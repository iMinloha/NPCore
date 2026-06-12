# =================================[CorePP — C++ Deep Learning Library]================================
cmake_minimum_required(VERSION 3.18)
project(CorePP
    VERSION 1.0.0
    DESCRIPTION "C++ Deep Learning Library with SIMD + CUDA acceleration"
    LANGUAGES CXX
)

# =================================[Options]================================
option(BUILD_SHARED_LIBS "Build CorePP as shared library (DLL)" OFF)
option(COREPP_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

# =================================[CUDA Detection & Setup]================================
if(COREPP_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

        # nvcc on Windows needs MSVC as host compiler
        if(WIN32 AND MINGW)
            # Find MSVC cl.exe for nvcc host compilation
            if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
                # Common VS 2022 paths
                foreach(_vs_path
                    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe"
                    "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe"
                    "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe"
                )
                    if(EXISTS "${_vs_path}")
                        set(CMAKE_CUDA_HOST_COMPILER "${_vs_path}" CACHE FILEPATH "MSVC compiler for nvcc")
                        break()
                    endif()
                endforeach()
                if(NOT CMAKE_CUDA_HOST_COMPILER)
                    message(WARNING "MSVC not found at standard paths. Set CMAKE_CUDA_HOST_COMPILER manually.")
                    message(WARNING "Example: -DCMAKE_CUDA_HOST_COMPILER=C:/path/to/cl.exe")
                endif()
            endif()
            message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
        endif()

        set(COREPP_CUDA_SOURCES
            CorePP/Cuda/cuda_device.cu
            CorePP/Cuda/cuda_gemm.cu
            CorePP/Cuda/cuda_elementwise.cu
            CorePP/Cuda/cuda_rnn.cu
        )
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    else()
        message(WARNING "CUDA Toolkit not found. Building CPU-only.")
        set(COREPP_ENABLE_CUDA OFF)
    endif()
endif()

# =================================[Compiler & Platform Settings]================================
include(cmake/CompilerSettings.cmake)

# =================================[Output Directories]================================
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =================================[Install Prefix]================================
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install prefix" FORCE)
endif()

# =================================[Source Files]================================
include(cmake/CorePPSources.cmake)

foreach(_src ${COREPP_ALL_SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
        message(FATAL_ERROR "Missing source file: ${_src}")
    endif()
endforeach()

# =================================[Library Target]================================
if(BUILD_SHARED_LIBS)
    add_library(CorePP SHARED
        ${COREPP_ALL_SOURCES}
        ${COREPP_CUDA_SOURCES}
    )
    target_compile_definitions(CorePP PRIVATE COREPP_BUILD_DLL)
    target_compile_definitions(CorePP PUBLIC  COREPP_USE_DLL)
else()
    add_library(CorePP STATIC
        ${COREPP_ALL_SOURCES}
        ${COREPP_CUDA_SOURCES}
    )
endif()

add_library(CorePP::CorePP ALIAS CorePP)

target_include_directories(CorePP PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CorePP>
    $<INSTALL_INTERFACE:include>
)

if(COREPP_ENABLE_CUDA)
    target_include_directories(CorePP PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/CorePP/Cuda)
    target_compile_definitions(CorePP PUBLIC COREPP_ENABLE_CUDA)
    target_link_libraries(CorePP PUBLIC CUDA::cudart)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(CorePP PUBLIC OpenMP::OpenMP_CXX)
endif()

# =================================[Install & Export]================================
include(cmake/InstallRules.cmake)

# =================================[Examples]================================
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
