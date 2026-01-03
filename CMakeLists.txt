cmake_minimum_required(VERSION 3.10)
project(vo_submission)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

add_executable(vo_submission src/vo_submission.cpp)
target_link_libraries(vo_submission ${OpenCV_LIBS})
