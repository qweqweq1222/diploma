cmake_minimum_required(VERSION 3.21)
project(start)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_library(lb 
  src/reader.cpp 
  header/algos.h
  header/final_tests.h 
  header/optim_struct.h 
  header/pointspreparation.h 
  header/reader.h
)

target_link_libraries(lb 
  PUBLIC opencv::core
  PUBLIC opencv::calib3d
  PUBLIC opencv::video
  PRIVATE opencv::imgcodecs
  PUBLIC opencv::features2d
  PUBLIC opencv::imgproc
  PUBLIC opencv::highgui
  PUBLIC Ceres::ceres
)
target_include_directories(lb PUBLIC header)

add_executable(start src/main.cpp)
target_link_libraries(start lb opencv::calib3d)

