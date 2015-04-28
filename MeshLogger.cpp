#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>
#include <SceneGraph/SimCam.h>
#include <calibu/Calibu.h>
#include <HAL/Utils/GetPot>
#include <HAL/Messages/Logger.h>

///////////////////////////////////////////////////////////////////////////
const char USAGE[] =
    "Usage:     MeshLogger -mesh <mesh file>  -cmod <camera rig file>  <options>\n"
    "\n"
    "where mesh can be of format: Blend, OBJ\n"
    "\n"
    "Options:\n"
    "   -sdir <source directory>    Directory where mesh and camera files are stored.\n"
    "   -poses <pose file>          File of poses for camera rig.\n"
    "   -depth                      Capture depth images.\n"
    "   -os                         Oversample.\n"
    "\n"
    "Example:\n"
    "MeshLogger  -mesh World.blend  -cmod cameras.xml  -depth\n";


///////////////////////////////////////////////////////////////////////////
inline void NormalizeDepth(
    float*       depth_ptr,
    unsigned int size
  )
{
  // Find max depth.
  float max_depth = 0;

  for (unsigned int ii = 0; ii < size; ii++) {
    if (max_depth < depth_ptr[ii]) {
      max_depth = depth_ptr[ii];
    }
  }

  if (max_depth == 0) {
    return;
  }

  // Normalize
  for (unsigned int ii = 0; ii < size; ii++) {
    depth_ptr[ii] = depth_ptr[ii] / max_depth;
  }
}


///////////////////////////////////////////////////////////////////////////
bool LoadPoseFile(
    const std::string&          file_name,
    std::vector<Sophus::SE3d>&  vec_poses
  )
{
  if (file_name.empty()) {
    return false;
  }

  std::ifstream   file;
  Eigen::Vector6d pose;

  file.open(file_name.c_str());

  if (file.is_open()) {
    std::cout << "MeshLogger: File opened..." << std::endl;

    // Read as T matrix.
    /*
        Eigen::Matrix4d T;
        file >> T(0,0) >> T(0,1) >> T(0,2) >> T(0,3) >>
                 T(1,0) >> T(1,1) >> T(1,2) >> T(1,3) >>
                 T(2,0) >> T(2,1) >> T(2,2) >> T(2,3) >>
                 T(3,0) >> T(3,1) >> T(3,2) >> T(3,3);
        pose = SceneGraph::GLT2Cart(Tt*T);
    */
    file >> pose(0) >> pose(1) >> pose(2) >> pose(3) >> pose(4) >> pose(5);

    while (!file.eof()) {
      vec_poses.push_back(Sophus::SE3d(SceneGraph::GLCart2T(pose)));

      file >> pose(0) >> pose(1)  >> pose(2) >> pose(3) >> pose(4) >> pose(5);

      // Read as T matrix.
      /*
          file >> T(0,0) >> T(0,1) >> T(0,2) >> T(0,3) >>
                   T(1,0) >> T(1,1) >> T(1,2) >> T(1,3) >>
                   T(2,0) >> T(2,1) >> T(2,2) >> T(2,3) >>
                   T(3,0) >> T(3,1) >> T(3,2) >> T(3,3);
          pose = SceneGraph::GLT2Cart(Tt*T);
      */
    }
  } else {
    std::cerr << "MeshLogger: Error opening pose file!" << std::endl;
    return false;
  }

  file.close();

  std::cout << "MeshLogger: Pose file closed." << std::endl;
  std::cout << "MeshLogger: " << vec_poses.size() << " poses read." << std::endl;
  return true;
}


/////////////////////////////////////////////////////////////////////////////
template<typename Type>
inline Type Interpolate(float           x,      // Input: X coordinate.
                        float           y,      // Input: Y coordinate.
                        const Type*     ptr,    // Input: Pointer to image.
                        unsigned int    width,  // Input: Image width.
                        unsigned int    height  // Input: Image height.
    )
{
  if (!((x >= 0) && (y >= 0) && (x <= width-2) && (y <= height-2))) {
    return 0;
  }

  x = std::max(std::min(x, static_cast<float>(width)-2.0f), 2.0f);
  y = std::max(std::min(y, static_cast<float>(height)-2.0f), 2.0f);

  const int    px  = static_cast<int>(x);  /* top-left corner */
  const int    py  = static_cast<int>(y);
  const float  ax  = x-px;
  const float  ay  = y-py;
  const float  ax1 = 1.0f-ax;
  const float  ay1 = 1.0f-ay;

  const Type* p0  = ptr+(width*py)+px;

  Type        p1  = p0[0];
  Type        p2  = p0[1];
  Type        p3  = p0[width];
  Type        p4  = p0[width+1];

  p1 *= ay1;
  p2 *= ay1;
  p3 *= ay;
  p4 *= ay;
  p1 += p3;
  p2 += p4;
  p1 *= ax1;
  p2 *= ax;

  return p1+p2;
}


///////////////////////////////////////////////////////////////////////////
inline double RFactorInv(double r, double w) {
  if (w * w < 1E-5) {
    return 1.0;
  } else {
    const double wby2 = w / 2.0;
    const double mul2tanwby2 = tan(wby2) * 2.0;
    if (r * r < 1E-5) {
      return w / mul2tanwby2;
    } else {
      return tan(r * w) / (r * mul2tanwby2);
    }
  }
}

///////////////////////////////////////////////////////////////////////////
template<typename Type>
void WarpImage(Type* in_ptr, Type* out_ptr, int width, int height,
               const Eigen::Matrix3d& K, double distortion)
{
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      Eigen::Vector2d p_distorted_ray;
      p_distorted_ray <<  (col - K(0,2)) / K(0,0),
                          (row - K(1,2)) / K(1,1);
      double factor = RFactorInv(p_distorted_ray.norm(), distortion);
      Eigen::Vector2d p_undistorted;
      p_undistorted(0) = ((col - K(0,2)) * factor) + K(0,2);
      p_undistorted(1) = ((row - K(1,2)) * factor) + K(1,2);

      const int idx = col + row*width;

      // Check if undistorted point falls inside image.
      if (p_undistorted(0) < 0 || p_undistorted(0) > width
       || p_undistorted(1) < 0 || p_undistorted(1) > height) {
        out_ptr[idx] = 0;
      } else {
        out_ptr[idx] = Interpolate<Type>(p_undistorted(0), p_undistorted(1),
                                         in_ptr, width, height);
      }
    }
  }
}



///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  ///-------------------- Parse Options
  if (argc < 1) {
    std::cout << USAGE << std::endl;
    exit(EXIT_FAILURE);
  }

  GetPot cl_args(argc, argv);

  if (cl_args.search(3, "--help", "-help", "-h")) {
    std::cout << USAGE << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string mesh_file     = cl_args.follow("", "-mesh");
  const std::string struct_file   = cl_args.follow("", "-struct");
  const std::string cam_mod_file  = cl_args.follow("", "-cmod");
  const std::string source_dir    = cl_args.follow(".", "-sdir");
  const std::string pose_file     = cl_args.follow("", "-poses");
  const bool        capture_depth = cl_args.search("-depth");
  const bool        oversample    = cl_args.search("-os");

  if (mesh_file.empty() || cam_mod_file.empty()) {
    std::cout << "error: Input files missing!" << std::endl;
    std::cout << USAGE << std::endl;
    exit(EXIT_FAILURE);
  }


  ///-------------------- Load Camera Rig
  std::shared_ptr<calibu::Rig<double>> rig =
          calibu::ReadXmlRig(source_dir + "/" + cam_mod_file);

  // TODO(jmf): Allow a different number of cameras (other than 2).
  const size_t num_cams = rig->NumCams();

  if (num_cams != 2) {
    std::cerr << "error: Two cameras are required to run this program." << std::endl;
    exit(EXIT_FAILURE);
  }

  // TODO(jmf): Poses and camera files are assumed/forced to be in robotics
  // reference frame. In the future, allow any reference frame by having
  // calibu reading the RDF from the file and setting things up appropriately.
  rig = calibu::ToCoordinateConvention(rig, calibu::RdfRobotics);

  std::shared_ptr<calibu::CameraInterface<double>> left_cam_mod =
      rig->cameras_[0];
  std::shared_ptr<calibu::CameraInterface<double>> right_cam_mod =
      rig->cameras_[1];

  if (oversample) {
    left_cam_mod->Scale(4);
    right_cam_mod->Scale(4);
  }

  // camera poses w.r.t. the rig
  Sophus::SE3d T_rig_cam1 = rig->cameras_[0]->Pose();
  Sophus::SE3d T_rig_cam2 = rig->cameras_[1]->Pose();

  // cameras are assumed to be identical in width and height as well
  // as having the same intrinsics
  // TODO(jmf): Allow different camera intrinsics?
  size_t img_width = left_cam_mod->Width();
  size_t img_height = left_cam_mod->Height();
  if (oversample) {
    img_width *= 4;
    img_height *= 4;
  }

  Eigen::Matrix3d K_left = left_cam_mod->K();
  Eigen::Matrix3d K_right = right_cam_mod->K();



  ///-------------------- Set up GUI
  // Create OpenGL window in single line thanks to GLUT
  pangolin::CreateGlutWindowAndBind("MeshLogger", 1280, 640);
  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();
  glewInit();

  // Scenegraph to hold GLObjects and relative transformations.
  SceneGraph::GLSceneGraph gl_graph;

  // Set up lighting (usually not required).
  //    SceneGraph::GLLight glLight;
  //    glLight.SetPosition(10,10,-100);
  //    glGraph.AddChild( &glLight );

  // Define grid object.
  SceneGraph::GLGrid gl_grid;
  gl_grid.SetPose(0, 0, 1, 0, 0, 0);
  gl_graph.AddChild(&gl_grid);

  // Set up mesh.
  SceneGraph::GLMesh gl_mesh;
  try {
    gl_mesh.Init(mesh_file);
    gl_mesh.SetPerceptable(true);
    gl_graph.AddChild(&gl_mesh);
    std::cout << "MeshLogger: Mesh '" << mesh_file << "' loaded." << std::endl;
  } catch(std::exception) {
    std::cerr << "error: Cannot load mesh. Check file exists." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Set up additional structure.
  if (struct_file.empty() ==  false) {
    SceneGraph::GLMesh gl_struct;
    try {
      gl_struct.Init(struct_file);
      gl_struct.SetPerceptable(true);
      gl_struct.SetScale(4.0);
      gl_struct.SetPose(0, 0, 1, -M_PI / 2, 0, 0);
      gl_graph.AddChild(&gl_struct);
      std::cout << "MeshLogger: Mesh '" << struct_file << "' loaded." << std::endl;
    } catch(std::exception) {
      std::cerr << "error: Cannot load structure file. Check file exists." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Set up axis for camera pose.
  SceneGraph::GLAxis gl_axis;
  gl_graph.AddChild(&gl_axis);

  ///-------------------- Load File Poses (if provided)
  std::vector<Sophus::SE3d> vec_poses;
  const bool have_pose_file = LoadPoseFile(pose_file, vec_poses);
  size_t pose_idx = 0;

  // Rig's world pose.
  Sophus::SE3d T_w_rig;

  // Rig's velocity (as per user input).
  Eigen::Vector6d rig_velocity;
  rig_velocity.setZero();

  if (have_pose_file) {
    T_w_rig = vec_poses[0];
  }


  ///-------------------- Initialize cameras
  SceneGraph::GLSimCam gl_cam_left;
  SceneGraph::GLSimCam gl_cam_right;

  if (capture_depth) {
    gl_cam_left.Init(&gl_graph, (T_w_rig * T_rig_cam1).matrix(), K_left, img_width,
                   img_height, SceneGraph::eSimCamLuminance|SceneGraph::eSimCamDepth);
  } else {
    gl_cam_left.Init(&gl_graph, (T_w_rig * T_rig_cam1).matrix(), K_left, img_width,
                   img_height, SceneGraph::eSimCamLuminance);
  }

  gl_cam_right.Init(&gl_graph, (T_w_rig * T_rig_cam2).matrix(), K_right, img_width,
                  img_height, SceneGraph::eSimCamLuminance);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState gl_state(pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 100000),
                                       pangolin::ModelViewLookAt(-6, 0, -30, 1, 0, 0, pangolin::AxisNegZ));

  // pangolinlin abstracts the OpenGL viewport as a View.
  // Here we get a reference to the default 'base' view.
  pangolin::View& gl_base_view = pangolin::DisplayBase();

  // We define a new view which will reside within the container.
  pangolin::View gl_view3D;

  // We set the views location on screen and add a handler which will
  // let user input update the model_view matrix (stacks3d) and feed through
  // to our scenegraph.
  gl_view3D.SetBounds(0.0, 1.0, 0.0, 3.0/4.0, 640.0f/480.0f);
  gl_view3D.SetHandler(new SceneGraph::HandlerSceneGraph(gl_graph, gl_state, pangolin::AxisNone));
  gl_view3D.SetDrawFunction(SceneGraph::ActivateDrawFunctor(gl_graph, gl_state));

  // display images
  SceneGraph::ImageView gl_left_img(true, true);
  gl_left_img.SetBounds(2.0/3.0, 1.0, 3.0/4.0, 1.0, static_cast<double>(img_width/img_height));

  SceneGraph::ImageView gl_right_img(true, true);
  gl_right_img.SetBounds(1.0/3.0, 2.0/3.0, 3.0/4.0, 1.0, static_cast<double>(img_width/img_height));

  SceneGraph::ImageView gl_depth_img(true, false);
  if (capture_depth) {
    gl_depth_img.SetBounds(0.0, 1.0/3.0, 3.0/4.0, 1.0, static_cast<double>(img_width/img_height));
  }

  // Add our views as children to the base container.
  gl_base_view.AddDisplay(gl_view3D);
  gl_base_view.AddDisplay(gl_left_img);
  gl_base_view.AddDisplay(gl_right_img);

  if (capture_depth) {
    gl_base_view.AddDisplay(gl_depth_img);
  }

  ///-------------------- Program control
  bool capture = have_pose_file ? true : false;
  bool pause   = false;

  // register key callbacks
  pangolin::RegisterKeyPressCallback('e', [&rig_velocity] { rig_velocity(0) += 0.01; });
  pangolin::RegisterKeyPressCallback('q', [&rig_velocity] { rig_velocity(0) -= 0.01; });
  pangolin::RegisterKeyPressCallback('a', [&rig_velocity] { rig_velocity(1) -= 0.01; });
  pangolin::RegisterKeyPressCallback('d', [&rig_velocity] { rig_velocity(1) += 0.01; });
  pangolin::RegisterKeyPressCallback('w', [&rig_velocity] { rig_velocity(2) -= 0.01; });
  pangolin::RegisterKeyPressCallback('s', [&rig_velocity] { rig_velocity(2) += 0.01; });
  pangolin::RegisterKeyPressCallback('u', [&rig_velocity] { rig_velocity(3) -= 0.005; });
  pangolin::RegisterKeyPressCallback('o', [&rig_velocity] { rig_velocity(3) += 0.005; });
  pangolin::RegisterKeyPressCallback('i', [&rig_velocity] { rig_velocity(4) += 0.005; });
  pangolin::RegisterKeyPressCallback('k', [&rig_velocity] { rig_velocity(4) -= 0.005; });
  pangolin::RegisterKeyPressCallback('j', [&rig_velocity] { rig_velocity(5) -= 0.005; });
  pangolin::RegisterKeyPressCallback('l', [&rig_velocity] { rig_velocity(5) += 0.005; });
  pangolin::RegisterKeyPressCallback(' ', [&rig_velocity] { rig_velocity.setZero(); });
  pangolin::RegisterKeyPressCallback('c', [&capture] { capture = !capture; });
  pangolin::RegisterKeyPressCallback('n', [&pause] { pause = !pause; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r', [&T_w_rig] { T_w_rig = Sophus::SE3d(); });


  // Buffer for our images and depth map.
  unsigned char*  ptr_img_buff   = (unsigned char*)malloc(img_width * img_height);
  unsigned char*  ptr_img_buff_T  = (unsigned char*)malloc(img_width * img_height);
  float*          ptr_depth_buff = (float*)malloc(img_width * img_height * 4);
  float*          ptr_depth_buff_T = (float*)malloc(img_width * img_height * 4);

  // Aux variables for logging.
  hal::Msg         msg;
  hal::CameraMsg*  ptr_cam_msg = nullptr;
  hal::Logger&     logger = hal::Logger::GetInstance();

  // Default hooks for exiting (Esc) and fullscreen (tab).
  for (unsigned frame_number = 0; !pangolin::ShouldQuit(); frame_number++) {
    // Clear whole screen.
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    // Move camera by user's input.
    Sophus::SE3d Tvel(SceneGraph::GLCart2T(rig_velocity));
    T_w_rig = T_w_rig * Tvel;

    gl_cam_left.SetPoseRobot((T_w_rig * T_rig_cam1).matrix());
    gl_cam_right.SetPoseRobot((T_w_rig * T_rig_cam2).matrix());

    // "Take picture".
    gl_cam_left.RenderToTexture();    // will render to texture, then copy texture to CPU memory
    gl_cam_right.RenderToTexture();    // will render to texture, then copy texture to CPU memory

    // Follow camera.
    gl_state.Follow((T_w_rig * T_rig_cam1).matrix());

    // Render cameras.
    gl_view3D.Activate(gl_state);
    gl_cam_left.DrawCamera();
    gl_cam_right.DrawCamera();

    // If capturing, prepare protobuf message.
    if (capture && !pause) {
      msg.Clear();
      msg.set_timestamp(frame_number);
      ptr_cam_msg = msg.mutable_camera();
    }

    // Show left image.
    if (gl_cam_left.CaptureGrey(ptr_img_buff)) {
//      WarpImage<unsigned char>(pBuffImgT, pBuffImg, img_width, img_height, K_left, 0.92);
      gl_left_img.SetImage(ptr_img_buff, img_width, img_height, GL_INTENSITY, GL_LUMINANCE, GL_UNSIGNED_BYTE);
      if (capture && !pause) {
        if (oversample) {
          cv::Mat downsampled;
          cv::Mat upsampled(cv::Size(img_width, img_height), CV_8U, ptr_img_buff);
          cv::pyrDown(upsampled, downsampled, cv::Size(img_width/2, img_height/2));
          cv::pyrDown(downsampled, downsampled, cv::Size(img_width/4, img_height/4));
          hal::ImageMsg* img_ptr = ptr_cam_msg->add_image();
          img_ptr->set_width(img_width/4);
          img_ptr->set_height(img_height/4);
          img_ptr->set_data(downsampled.data, img_width * img_height / 16);
          img_ptr->set_type(hal::PB_UNSIGNED_BYTE);
          img_ptr->set_format(hal::PB_LUMINANCE);
        } else {
          hal::ImageMsg* img_ptr = ptr_cam_msg->add_image();
          img_ptr->set_width(img_width);
          img_ptr->set_height(img_height);
          img_ptr->set_data(ptr_img_buff, img_width * img_height);
          img_ptr->set_type(hal::PB_UNSIGNED_BYTE);
          img_ptr->set_format(hal::PB_LUMINANCE);
        }
      }
    }

    // Show right image.
#if 1
    if (gl_cam_right.CaptureGrey(ptr_img_buff)) {
      gl_right_img.SetImage(ptr_img_buff, img_width, img_height, GL_INTENSITY, GL_LUMINANCE, GL_UNSIGNED_BYTE);
      if (capture && !pause) {
        if (oversample) {
          cv::Mat downsampled;
          cv::Mat upsampled(cv::Size(img_width, img_height), CV_8U, ptr_img_buff);
          cv::pyrDown(upsampled, downsampled, cv::Size(img_width/2, img_height/2));
          cv::pyrDown(downsampled, downsampled, cv::Size(img_width/4, img_height/4));
          hal::ImageMsg* img_ptr = ptr_cam_msg->add_image();
          img_ptr->set_width(img_width/4);
          img_ptr->set_height(img_height/4);
          img_ptr->set_data(downsampled.data, img_width*img_height/16);
          img_ptr->set_type(hal::PB_UNSIGNED_BYTE);
          img_ptr->set_format(hal::PB_LUMINANCE);
        } else {
          hal::ImageMsg* img_ptr = ptr_cam_msg->add_image();
          img_ptr->set_width(img_width);
          img_ptr->set_height(img_height);
          img_ptr->set_data(ptr_img_buff, img_width * img_height);
          img_ptr->set_type(hal::PB_UNSIGNED_BYTE);
          img_ptr->set_format(hal::PB_LUMINANCE);
        }
      }
    }
#endif

    // Show depth.
    if (capture_depth) {
      if (gl_cam_left.CaptureDepth(ptr_depth_buff)) {
//        WarpImage<float>(pBuffDepthT, pBuffDepth, img_width, img_height, K_left, 0.92);
        if (capture && !pause) {
          if (oversample) {
            cv::Mat downsampled;
            cv::Mat upsampled(cv::Size(img_width, img_height), CV_32F, ptr_depth_buff);
            cv::pyrDown(upsampled, downsampled, cv::Size(img_width/2, img_height/2));
            cv::pyrDown(downsampled, downsampled, cv::Size(img_width/4, img_height/4));
            hal::ImageMsg* img_ptr = ptr_cam_msg->add_image();
            img_ptr->set_width(img_width/4);
            img_ptr->set_height(img_height/4);
            img_ptr->set_data(downsampled.data, img_width*img_height*4/16);
            img_ptr->set_type(hal::PB_FLOAT);
            img_ptr->set_format(hal::PB_LUMINANCE);
          } else {
            hal::ImageMsg* img_ptr = ptr_cam_msg->add_image();
            img_ptr->set_width(img_width);
            img_ptr->set_height(img_height);
            img_ptr->set_data(ptr_depth_buff, img_width*img_height*4);
            img_ptr->set_type(hal::PB_FLOAT);
            img_ptr->set_format(hal::PB_LUMINANCE);
          }
        }
        NormalizeDepth(ptr_depth_buff, img_width * img_height);
        gl_depth_img.SetImage(ptr_depth_buff, img_width, img_height, GL_INTENSITY, GL_LUMINANCE, GL_FLOAT);
      }
    }

    // Log message.
    if (capture && !pause) {
      logger.LogMessage(msg);
#if 0
      // This is to change endianness for Capri.
      cv::Mat grey(img_height, img_width, CV_8UC1, pBuffImgT);
      cv::Mat depth(img_height, img_width, CV_32FC1, pBuffDepthT);
      depth = depth * 1000;
      cv::Mat depth_mm;
      depth.convertTo(depth_mm, CV_16UC1);
      cv::Mat depth_mm2(img_height, img_width, CV_16UC1);
      for (size_t ii = 0; ii < img_height; ++ii) {
        for (size_t jj = 0; jj < 2*img_width; ++jj) {
          if (jj % 2 == 0)
            depth_mm2.at<unsigned char>(ii,jj) = depth_mm.at<unsigned char>(ii,jj+1);
          else
            depth_mm2.at<unsigned char>(ii,jj) = depth_mm.at<unsigned char>(ii,jj-1);
        }
      }
      cv::imwrite("capri.pgm", grey);
      cv::imwrite("capri-depth.pgm", depth_mm2);
#endif
    }

    // Reset capture flag.
    if (capture) {
      capture = false;
    }

    // If we are rendering from an input file, set capture flag and reposition camera.
    if (have_pose_file && !pause) {
      capture = true;
      pose_idx++;
      if (pose_idx >= vec_poses.size()) {
        std::cout << "MeshLogger: Finished rendering input file." << std::endl;
        break;
      }
      T_w_rig = vec_poses[pose_idx];
      gl_axis.SetPose(T_w_rig.matrix());
    }

    // Swap frames and process events.
    pangolin::FinishGlutFrame();

    // Pause for 1/60th of a second.
    usleep(1E6 / 60);
  }

  // Wait for logger to finish copying to disk.
  std::cout << "MeshLogger: Waiting for logger to flush queue... ";
  fflush(stdout);
  logger.StopLogging();
  std::cout << "Done!" << std::endl;

  return 0;
}
