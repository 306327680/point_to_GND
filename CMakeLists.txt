cmake_minimum_required(VERSION 3.14)
project(untitled)
find_package( Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
find_package(PCL REQUIRED QUIET)
include_directories(
        ${PCL_INCLUDE_DIRS}
)
set(CMAKE_CXX_STANDARD 14)
include_directories(
        ${EIGEN3_INCLUDE_DIR})
add_executable(untitled main.cpp)
target_link_libraries(untitled ${CERES_LIBRARIES}  ${PCL_LIBRARIES})