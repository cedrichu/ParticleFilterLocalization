cmake_minimum_required(VERSION 2.6)

project(hw_04)

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

find_package(OpenCV)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})

#set(SRCS src/main.cpp data/map/bee-map.c)
set(SRCS main.cpp)

# C++11
set_source_files_properties(${SRCS} PROPERTIES COMPILE_FLAGS -std=c++0x)

#set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath, -g . -L${CMAKE_CURRENT_BINARY_DIR}")

add_executable(hw_04 ${SRCS})

target_link_libraries(hw_04 ${OpenCV_LIBRARIES})

