project(VO)
cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(G2O REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(CSparse REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} demo.cpp)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)
