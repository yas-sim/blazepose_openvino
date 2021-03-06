# CMakeLists.txt - blazepose

cmake_minimum_required (VERSION 3.0.0)

project(blazepose)

set(TARGET_NAME blazepose)     # name of executable file
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIE")

find_package(InferenceEngine 1.1 REQUIRED)
find_package(OpenCV REQUIRED)
add_definitions(-DUSE_OPENCV)

include_directories( ${InferenceEngine_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
link_directories( )

add_executable( ${TARGET_NAME} main.cpp )    # list of source file(s)
set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} )
