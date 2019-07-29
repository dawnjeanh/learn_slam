project(g2o_demo)
cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

find_package(Boost COMPONENTS filesystem system iostreams REQUIRED)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(G2O REQUIRED)

include_directories( ${Boost_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS} )

add_executable(g2o_demo demo.cpp)

target_link_libraries( g2o_demo ${Boost_LIBRARIES} g2o_core g2o_stuff)
