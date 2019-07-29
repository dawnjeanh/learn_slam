project(orb)
cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

find_package(OpenCV 3.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(orb orb.cpp)

link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(orb ${OpenCV_LIBS})
