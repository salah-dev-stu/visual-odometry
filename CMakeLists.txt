cmake_minimum_required(VERSION 3.10)
project(vo_submission)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# PoseLib
set(POSELIB_DIR ${CMAKE_SOURCE_DIR}/poselib_src)
set(POSELIB_INCLUDE ${POSELIB_DIR})
set(POSELIB_LIB ${POSELIB_DIR}/build/PoseLib/libPoseLib.a)

add_executable(vo_submission src/vo_submission.cpp)
target_include_directories(vo_submission PRIVATE
    ${POSELIB_INCLUDE}
    ${EIGEN3_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(vo_submission ${OpenCV_LIBS} ${POSELIB_LIB})

# Trajectory Viewer with Pangolin
find_package(Pangolin QUIET)
if(Pangolin_FOUND)
    add_executable(trajectory_viewer src/trajectory_viewer.cpp)
    target_link_libraries(trajectory_viewer ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
    target_include_directories(trajectory_viewer PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
