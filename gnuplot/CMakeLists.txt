project(gnuplot_demo)
cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

find_package(Boost COMPONENTS filesystem system iostreams REQUIRED)
include_directories( ${Boost_INCLUDE_DIRS} )

add_executable(gnuplot_demo demo.cpp)

target_link_libraries( gnuplot_demo ${Boost_LIBRARIES} )
