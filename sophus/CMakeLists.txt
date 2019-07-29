project(sophus_demo)
cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

include_directories("/usr/include/eigen3")
add_executable(sophus_demo demo.cpp)

target_link_libraries( sophus_demo ${Sophus_LIBRARIES} )
