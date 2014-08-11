#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>
#include <SceneGraph/SimCam.h>
#include <calibu/Calibu.h>
#include <HAL/Utils/GetPot>
#include <PbMsgs/Logger.h>

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

  Eigen::Matrix4d T;
  Eigen::Matrix4d Tt = SceneGraph::GLCart2T(0, 0, 0, 0, 0, -M_PI/2.0);

  if (file.is_open()) {
    std::cout << "MeshLogger: File opened..." << std::endl;

    /*
        pFile >> T(0,0) >> T(0,1) >> T(0,2) >> T(0,3) >>
                 T(1,0) >> T(1,1) >> T(1,2) >> T(1,3) >>
                 T(2,0) >> T(2,1) >> T(2,2) >> T(2,3) >>
                 T(3,0) >> T(3,1) >> T(3,2) >> T(3,3);
        Pose = SceneGraph::GLT2Cart(T);
        */
    file >> pose(0) >> pose(1) >> pose(2) >> pose(3) >> pose(4) >> pose(5);

    while (!file.eof()) {
      /*
          vPoses.push_back( Sophus::SE3d( Tt.inverse()*T*Tt ) );
          pFile >> T(0,0) >> T(0,1) >> T(0,2) >> T(0,3) >>
                   T(1,0) >> T(1,1) >> T(1,2) >> T(1,3) >>
                   T(2,0) >> T(2,1) >> T(2,2) >> T(2,3) >>
                   T(3,0) >> T(3,1) >> T(3,2) >> T(3,3);
          Pose = SceneGraph::GLT2Cart(T);
         */
      vec_poses.push_back(Sophus::SE3d(SceneGraph::GLCart2T(pose)));
      file >> pose(0) >> pose(1)  >> pose(2) >> pose(3) >> pose(4) >> pose(5);
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

  GetPot clArgs(argc, argv);

  if (clArgs.search(3, "--help", "-help", "-h")) {
    std::cout << USAGE << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string mesh_file     = clArgs.follow("", "-mesh");
  const std::string struct_file    = clArgs.follow("", "-struct");
  const std::string cam_mod_file  = clArgs.follow("", "-cmod");
  const std::string source_dir    = clArgs.follow(".", "-sdir");
  const std::string pose_file     = clArgs.follow("", "-poses");
  const bool        capture_depth = clArgs.search("-depth");
  const bool        oversample    = clArgs.search("-os");

  if (mesh_file.empty() || cam_mod_file.empty()) {
    std::cout << "error: Input files missing!" << std::endl;
    std::cout << USAGE << std::endl;
    exit(EXIT_FAILURE);
  }


  ///-------------------- Load Camera Rig
  calibu::CameraRig rig = calibu::ReadXmlRig(source_dir + "/" + cam_mod_file);

  // TODO(jmf): Allow a different number of cameras (other than 2).
  const size_t num_cams = rig.cameras.size();

  if (num_cams != 2) {
    std::cerr << "error: Two cameras are required to run this program." << std::endl;
    exit(EXIT_FAILURE);
  }

  // TODO(jmf): Poses and camera files are assumed/forced to be in robotics
  // reference frame. In the future, allow any reference frame by having
  // calibu reading the RDF from the file and setting things up appropriately.
  rig = calibu::ToCoordinateConvention(rig, calibu::RdfRobotics);

  calibu::CameraModel left_cam_mod = rig.cameras[0].camera;
  calibu::CameraModel right_cam_mod = rig.cameras[1].camera;

  if (oversample) {
    left_cam_mod.Scale(4);
    right_cam_mod.Scale(4);
  }

  // camera poses w.r.t. the world
  Sophus::SE3d T_rig_cam1 = rig.cameras[0].T_wc;
  Sophus::SE3d T_rig_cam2 = rig.cameras[1].T_wc;

  // cameras are assumed to be identical in width and height as well
  // as having the same intrinsics
  // TODO(jmf): Allow different camera intrinsics?
  const size_t img_width = left_cam_mod.Width();
  const size_t img_height = left_cam_mod.Height();

  Eigen::Matrix3d K_left = left_cam_mod.K();
  Eigen::Matrix3d K_right = right_cam_mod.K();



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
//    gl_mesh.SetScale(2.0);
//    gl_mesh.SetPose(0, 0, 7, -M_PI / 2, 0, 0); // Monterey
    gl_mesh.SetPose(-53, -171, 1, -M_PI/2, 0, (-55*M_PI)/180.0); // Alabama House
    gl_graph.AddChild(&gl_mesh);
    std::cout << "MeshLogger: Mesh '" << mesh_file << "' loaded." << std::endl;
  } catch(std::exception) {
    std::cerr << "error: Cannot load mesh. Check file exists." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Set up mesh.
  SceneGraph::GLMesh gl_struct;
  try {
    gl_struct.Init(struct_file);
    gl_struct.SetPerceptable(true);
    gl_struct.SetScale(4.0);
    gl_struct.SetPose(0, 0, 1, -M_PI / 2, 0, 0);
    gl_graph.AddChild(&gl_struct);
    std::cout << "MeshLogger: Mesh '" << struct_file << "' loaded." << std::endl;
  } catch(std::exception) {
    std::cerr << "error: Cannot load mesh. Check file exists." << std::endl;
    exit(EXIT_FAILURE);
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
  SceneGraph::GLSimCam glCamLeft;
  SceneGraph::GLSimCam glCamRight;

  if (capture_depth) {
    glCamLeft.Init(&gl_graph, (T_w_rig * T_rig_cam1).matrix(), K_left, img_width,
                   img_height, SceneGraph::eSimCamLuminance|SceneGraph::eSimCamDepth);
  } else {
    glCamLeft.Init(&gl_graph, (T_w_rig * T_rig_cam1).matrix(), K_left, img_width,
                   img_height, SceneGraph::eSimCamLuminance);
  }

  glCamRight.Init(&gl_graph, (T_w_rig * T_rig_cam2).matrix(), K_right, img_width,
                  img_height, SceneGraph::eSimCamLuminance);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState glState(pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 100000),
                                      pangolin::ModelViewLookAt(-6, 0, -30, 1, 0, 0, pangolin::AxisNegZ));

  // pangolinlin abstracts the OpenGL viewport as a View.
  // Here we get a reference to the default 'base' view.
  pangolin::View& glBaseView = pangolin::DisplayBase();

  // We define a new view which will reside within the container.
  pangolin::View glView3D;

  // We set the views location on screen and add a handler which will
  // let user input update the model_view matrix (stacks3d) and feed through
  // to our scenegraph.
  glView3D.SetBounds(0.0, 1.0, 0.0, 3.0/4.0, 640.0f/480.0f);
  glView3D.SetHandler(new SceneGraph::HandlerSceneGraph(gl_graph, glState, pangolin::AxisNone));
  glView3D.SetDrawFunction(SceneGraph::ActivateDrawFunctor(gl_graph, glState));

  // display images
  SceneGraph::ImageView glLeftImg(true, true);
  glLeftImg.SetBounds(2.0/3.0, 1.0, 3.0/4.0, 1.0, static_cast<double>(img_width/img_height));

  SceneGraph::ImageView glRightImg(true, true);
  glRightImg.SetBounds(1.0/3.0, 2.0/3.0, 3.0/4.0, 1.0, static_cast<double>(img_width/img_height));

  SceneGraph::ImageView glDepthImg(true, false);
  if (capture_depth) {
    glDepthImg.SetBounds(0.0, 1.0/3.0, 3.0/4.0, 1.0, static_cast<double>(img_width/img_height));
  }

  // Add our views as children to the base container.
  glBaseView.AddDisplay(glView3D);
  glBaseView.AddDisplay(glLeftImg);
  glBaseView.AddDisplay(glRightImg);

  if (capture_depth) {
    glBaseView.AddDisplay(glDepthImg);
  }

  ///-------------------- Program control
  bool capture = false;
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
  unsigned char*  pBuffImg   = (unsigned char*)malloc(img_width * img_height);
  unsigned char*  pBuffImgT  = (unsigned char*)malloc(img_width * img_height);
  float*          pBuffDepth = (float*)malloc(img_width * img_height * 4);
  float*          pBuffDepthT = (float*)malloc(img_width * img_height * 4);

  // Aux variables for logging.
  pb::Msg         pbMsg;
  pb::CameraMsg*  pCamMsg = nullptr;

  // Default hooks for exiting (Esc) and fullscreen (tab).
  for (unsigned frame_number = 0; !pangolin::ShouldQuit(); frame_number++) {
    // Clear whole screen.
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    // Move camera by user's input.
    Sophus::SE3d Tvel(SceneGraph::GLCart2T(rig_velocity));
    T_w_rig = T_w_rig * Tvel;

    glCamLeft.SetPoseRobot((T_w_rig * T_rig_cam1).matrix());
    glCamRight.SetPoseRobot((T_w_rig * T_rig_cam2).matrix());

    // "Take picture".
    glCamLeft.RenderToTexture();    // will render to texture, then copy texture to CPU memory
    glCamRight.RenderToTexture();    // will render to texture, then copy texture to CPU memory

    // Follow camera.
    glState.Follow((T_w_rig * T_rig_cam1).matrix());

    // Render cameras.
    glView3D.Activate(glState);
    glCamLeft.DrawCamera();
    glCamRight.DrawCamera();

    // If capturing, prepare protobuf message.
    if (capture && !pause) {
      pbMsg.Clear();
      pbMsg.set_timestamp(frame_number);
      pCamMsg = pbMsg.mutable_camera();
    }

    // Show left image.
    if (glCamLeft.CaptureGrey(pBuffImg)) {
//      WarpImage<unsigned char>(pBuffImgT, pBuffImg, img_width, img_height, K_left, 0.92);
      glLeftImg.SetImage(pBuffImg, img_width, img_height, GL_INTENSITY, GL_LUMINANCE, GL_UNSIGNED_BYTE);
      if (capture && !pause) {
        if (oversample) {
          cv::Mat downsampled;
          cv::Mat upsampled(cv::Size(img_width, img_height), CV_8U, pBuffImg);
          cv::pyrDown(upsampled, downsampled, cv::Size(img_width/2, img_height/2));
          cv::pyrDown(downsampled, downsampled, cv::Size(img_width/4, img_height/4));
          pb::ImageMsg* pImg = pCamMsg->add_image();
          pImg->set_width(img_width/4);
          pImg->set_height(img_height/4);
          pImg->set_data(downsampled.data, img_width * img_height / 16);
          pImg->set_type(pb::PB_UNSIGNED_BYTE);
          pImg->set_format(pb::PB_LUMINANCE);
        } else {
          pb::ImageMsg* pImg = pCamMsg->add_image();
          pImg->set_width(img_width);
          pImg->set_height(img_height);
          pImg->set_data(pBuffImg, img_width * img_height);
          pImg->set_type(pb::PB_UNSIGNED_BYTE);
          pImg->set_format(pb::PB_LUMINANCE);
        }
      }
    }

    // Show right image.
#if 1
    if (glCamRight.CaptureGrey(pBuffImg)) {
      glRightImg.SetImage(pBuffImg, img_width, img_height, GL_INTENSITY, GL_LUMINANCE, GL_UNSIGNED_BYTE);
      if (capture && !pause) {
        if (oversample) {
          cv::Mat downsampled;
          cv::Mat upsampled(cv::Size(img_width, img_height), CV_8U, pBuffImg);
          cv::pyrDown(upsampled, downsampled, cv::Size(img_width/2, img_height/2));
          cv::pyrDown(downsampled, downsampled, cv::Size(img_width/4, img_height/4));
          pb::ImageMsg* pImg = pCamMsg->add_image();
          pImg->set_width(img_width/4);
          pImg->set_height(img_height/4);
          pImg->set_data(downsampled.data, img_width*img_height/16);
          pImg->set_type(pb::PB_UNSIGNED_BYTE);
          pImg->set_format(pb::PB_LUMINANCE);
        } else {
          pb::ImageMsg* pImg = pCamMsg->add_image();
          pImg->set_width(img_width);
          pImg->set_height(img_height);
          pImg->set_data(pBuffImg, img_width * img_height);
          pImg->set_type(pb::PB_UNSIGNED_BYTE);
          pImg->set_format(pb::PB_LUMINANCE);
        }
      }
    }
#endif

    // Show depth.
    if (capture_depth) {
      if (glCamLeft.CaptureDepth(pBuffDepth)) {
//        WarpImage<float>(pBuffDepthT, pBuffDepth, img_width, img_height, K_left, 0.92);
        if (capture && !pause) {
          if (oversample) {
            cv::Mat downsampled;
            cv::Mat upsampled(cv::Size(img_width, img_height), CV_32F, pBuffDepth);
            cv::pyrDown(upsampled, downsampled, cv::Size(img_width/2, img_height/2));
            cv::pyrDown(downsampled, downsampled, cv::Size(img_width/4, img_height/4));
            pb::ImageMsg* pImg = pCamMsg->add_image();
            pImg->set_width(img_width/4);
            pImg->set_height(img_height/4);
            pImg->set_data(downsampled.data, img_width*img_height*4/16);
            pImg->set_type(pb::PB_FLOAT);
            pImg->set_format(pb::PB_LUMINANCE);
          } else {
            pb::ImageMsg* pImg = pCamMsg->add_image();
            pImg->set_width(img_width);
            pImg->set_height(img_height);
            pImg->set_data(pBuffDepth, img_width*img_height*4);
            pImg->set_type(pb::PB_FLOAT);
            pImg->set_format(pb::PB_LUMINANCE);
          }
        }
        NormalizeDepth(pBuffDepth, img_width * img_height);
        glDepthImg.SetImage(pBuffDepth, img_width, img_height, GL_INTENSITY, GL_LUMINANCE, GL_FLOAT);
      }
    }

    // Log message.
    if (capture && !pause) {
      pb::Logger::GetInstance().LogMessage(pbMsg);
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
        exit(EXIT_SUCCESS);
      }
      T_w_rig = vec_poses[pose_idx];
      gl_axis.SetPose(T_w_rig.matrix());
    }

    // Swap frames and process events.
    pangolin::FinishGlutFrame();

    // Pause for 1/60th of a second.
    usleep(1E6 / 60);
  }

  return 0;
}
