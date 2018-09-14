cmake_minimum_required(VERSION 2.8)

project(flame)
set(${PROJECT_NAME}_VERSION 1.0.0)

add_definitions("-std=c++11")

# Locate external dependencies
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)
find_package(Boost 1.54 REQUIRED COMPONENTS system filesystem graph)
find_package(Sophus REQUIRED)

# Project includes
include_directories(SYSTEM
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        src)

# Build the library
add_subdirectory(src)
