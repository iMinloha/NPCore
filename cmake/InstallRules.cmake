# =================================[NPCore Install & Export Rules]================================

# --- Install headers into include/NPCore/ (standard namespaced layout) ---
# Users write: #include <NPCore/NPCore.h>
install(
    DIRECTORY NPCore/
    DESTINATION include/NPCore
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.inl"
    PATTERN "test" EXCLUDE
    PATTERN "Cuda" EXCLUDE
)

# --- Install library target ---
install(
    TARGETS NPCore
    EXPORT NPCoreTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# --- Generate and install package config ---
include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/NPCoreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NPCoreConfig.cmake
    INSTALL_DESTINATION lib/cmake/NPCore
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/NPCoreConfig.cmake
    DESTINATION lib/cmake/NPCore
)

install(
    EXPORT NPCoreTargets
    FILE NPCoreTargets.cmake
    NAMESPACE NPCore::
    DESTINATION lib/cmake/NPCore
)

# --- Create install directory at build time ---
add_custom_target(create_install_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}
    COMMENT "Creating install directory: ${CMAKE_INSTALL_PREFIX}"
)
