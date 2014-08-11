cmake_minimum_required( VERSION 2.8 )

project(MeshLogger)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

find_package(Pangolin 0.1 REQUIRED)
find_package(SceneGraph 0.1 REQUIRED)
find_package(HAL 0.1 REQUIRED)
find_package(PbMsgs 0.1 REQUIRED)
find_package(MINIGLOG REQUIRED)
find_package(Calibu 0.1 REQUIRED)
find_package(OpenCV2 REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${SceneGraph_INCLUDE_DIRS})
include_directories(${HAL_INCLUDE_DIRS})
include_directories(${Calibu_INCLUDE_DIRS})
include_directories(${OpenCV2_INCLUDE_DIRS})

link_directories(${HAL_LINK_DIRS})
link_directories(${Pangolin_LIBRARY_DIRS})

list(APPEND HDRS)

list(APPEND SRCS
    MeshLogger.cpp
)

add_executable(MeshLogger ${HDRS} ${SRCS})

target_link_libraries(MeshLogger
                        ${Pangolin_LIBRARIES}
                        ${SceneGraph_LIBRARIES}
                        ${HAL_LIBRARIES}
                        ${Calibu_LIBRARY}
                        ${OpenCV2_LIBRARIES}
                     )
