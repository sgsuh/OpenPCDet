cmake_minimum_required(VERSION 3.27.0)
project(pointpillar LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE
"Release")

find_package(CUDAToolkit REQUIRED)
# find_package(PCL REQUIRED COMPONENTS common io kdtree filters search visualization)
# find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
# find_package(OpenCV REQUIRED PATHS /usr/local/lib/cmake/opencv4)

set(CMAKE_CXX_FLAGS_RELEASE
"-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CUDA_FLAGS_RELEASE
"-Xcompiler -std=c++17,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CMAKE_CUDA_ARCHITECTURES 89)

set(TENSORRT_INCLUDE_DIRS
/usr/include/x86_64-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS
/usr/lib/x86_64-linux-gnu/)

include_directories(${CUDAToolkit_INCLUDE_DIRS}
${TENSORRT_INCLUDE_DIRS}
# ${PCL_INCLUDE_DIRS}
# ${EIGEN3_INCLUDE_DIRS}
# ${OpenCV_INCLUDE_DIRS}
include)

link_directories(${CUDAToolkit_LIBRARY_DIR}
${TENSORRT_LIBRARY_DIRS}
# ${PCL_LIBRARY_DIRS}
)

# add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE
CORE_FILES
src/*.cu
src/*.cpp)

add_library(pointpillar_core
SHARED
${CORE_FILES})

target_link_libraries(pointpillar_core
CUDA::cudart
libnvinfer.so)

add_executable(pointpillar
apps/main.cpp)

target_link_libraries(pointpillar
pointpillar_core
yaml-cpp)