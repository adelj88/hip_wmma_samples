cmake_minimum_required(VERSION 3.10)
project(hip_wmma_samples LANGUAGES CXX)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Find HIP package (make sure HIP is installed and the path is set correctly)
if(WIN32)
  set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
  set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${ROCM_ROOT}/lib/cmake/hip
  ${ROCM_ROOT}/hip/cmake # FindHIP.cmake
)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_ROOT}/llvm ${ROCM_ROOT} ${ROCM_ROOT}/hip)
find_package(hip REQUIRED CONFIG PATHS ${ROCM_ROOT} /opt/rocm)
find_package(rocBLAS REQUIRED)

include(cmake/Dependencies.cmake)

# Add subdirectories
add_subdirectory(hgemm)

