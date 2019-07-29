project(pcl_demo)
cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

find_package( OpenCV 3.4 REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

include_directories("/usr/include/eigen3")

add_executable(pcl_demo demo.cpp)

target_link_libraries(pcl_demo ${OpenCV_LIBS} ${PCL_LIBRARIES})