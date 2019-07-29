project(eigen_demo)
cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

include_directories("/usr/include/eigen3")
add_executable(eigen_demo demo.cpp)