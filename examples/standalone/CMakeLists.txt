cmake_minimum_required(VERSION 3.18)
project(NPCoreDemo LANGUAGES CXX)

set(NPCORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/install")

# ---- Verify install/ exists ----
if(NOT EXISTS "${NPCORE_ROOT}/include/NPCore/NPCore.h")
    message(FATAL_ERROR
        "install/ not found. Copy NPCore's install directory here.\n"
        "  cp -r <npcore>/cmake-build-debug/install .")
endif()

# ---- NPCore headers ----
# include/NPCore/ mirrors the source tree layout so that
# internal relative includes ("Core/Matrix.hpp", "Layers/Module.h", …)
# resolve exactly as they do during a normal in-tree build.
set(NPCORE_INCLUDE "${NPCORE_ROOT}/include/NPCore")

# ---- NPCore library ----
if(WIN32 AND EXISTS "${NPCORE_ROOT}/bin/libNPCore.dll")
    # Shared library (DLL)
    set(NPCORE_LIB "${NPCORE_ROOT}/bin/libNPCore.dll")
    set(NPCORE_IMPLIB "${NPCORE_ROOT}/lib/libNPCore.dll.a")
    add_library(npcore SHARED IMPORTED)
    set_target_properties(npcore PROPERTIES
        IMPORTED_LOCATION "${NPCORE_LIB}"
        IMPORTED_IMPLIB "${NPCORE_IMPLIB}"
    )
elseif(EXISTS "${NPCORE_ROOT}/lib/libNPCore.a")
    # Static library
    add_library(npcore STATIC IMPORTED)
    set_target_properties(npcore PROPERTIES
        IMPORTED_LOCATION "${NPCORE_ROOT}/lib/libNPCore.a"
    )
else()
    message(FATAL_ERROR "NPCore library not found in install/lib/")
endif()

target_include_directories(npcore INTERFACE "${NPCORE_INCLUDE}")

# ---- OpenMP (required by NPCore) ----
find_package(OpenMP REQUIRED)

# ---- Build demos ----
function(add_npcore_demo name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE npcore OpenMP::OpenMP_CXX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${name} PRIVATE -mavx2 -mfma)
        target_link_options(${name} PRIVATE -fopenmp)
    endif()
    # Copy DLL beside executable for shared builds
    if(WIN32 AND NPCORE_LIB)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${NPCORE_LIB}" "$<TARGET_FILE_DIR:${name}>"
            COMMENT "Copying NPCore DLL for ${name}")
    endif()
endfunction()

add_npcore_demo(demo_app          demo.cpp)
add_npcore_demo(demo_classifier   demo_classifier.cpp)
