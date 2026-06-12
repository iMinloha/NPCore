# =================================[CorePP Dynamic Linking Example]================================
# Standalone CMake project — links against pre-built CorePP shared library.
#
# Build:
#   cmake -B _build -DCOREPP_ROOT=../.. && cmake --build _build
#
# Or manually:
#   g++ -std=c++20 -I../../CorePP main.cpp -L../../_build/lib -lCorePP -fopenmp

cmake_minimum_required(VERSION 3.15)
project(CorePPExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Find CorePP (point COREPP_ROOT to the project root)
if(NOT DEFINED COREPP_ROOT)
    set(COREPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

# Include dirs
set(COREPP_INCLUDE_DIR "${COREPP_ROOT}/CorePP")

# Find the shared library
find_library(COREPP_LIBRARY CorePP
    PATHS "${COREPP_ROOT}/_build/lib"
          "${COREPP_ROOT}/build/lib"
    NO_DEFAULT_PATH
)

if(NOT COREPP_LIBRARY)
    message(FATAL_ERROR "CorePP library not found. Build CorePP first with BUILD_SHARED_LIBS=ON")
endif()

message(STATUS "CorePP library: ${COREPP_LIBRARY}")
message(STATUS "CorePP include: ${COREPP_INCLUDE_DIR}")

# Example executable
add_executable(example_basic main.cpp)
target_include_directories(example_basic PRIVATE ${COREPP_INCLUDE_DIR})
target_link_libraries(example_basic PRIVATE ${COREPP_LIBRARY})

# Find OpenMP (CorePP needs it)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(example_basic PRIVATE OpenMP::OpenMP_CXX)
endif()

# Copy DLL to output dir on Windows
if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(TARGET example_basic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:CorePP>"
        "$<TARGET_FILE_DIR:example_basic>"
    )
endif()
