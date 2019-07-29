project(ceres_demo)
cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

find_package(Boost COMPONENTS filesystem system iostreams REQUIRED)
include_directories( ${Boost_INCLUDE_DIRS} )

find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS} )

add_executable(ceres_demo demo.cpp)

target_link_libraries( ceres_demo ${Boost_LIBRARIES} ${CERES_LIBRARIES})
