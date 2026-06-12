# =================================[CorePP Install & Export Rules]================================

# --- Install headers (preserve directory structure) ---
install(
    DIRECTORY CorePP/
    DESTINATION include/CorePP
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.inl"
    PATTERN "test" EXCLUDE
)

# --- Install library target ---
install(
    TARGETS CorePP
    EXPORT CorePPTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# --- Generate and install package config ---
include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/CorePPConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CorePPConfig.cmake
    INSTALL_DESTINATION lib/cmake/CorePP
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/CorePPConfig.cmake
    DESTINATION lib/cmake/CorePP
)

install(
    EXPORT CorePPTargets
    FILE CorePPTargets.cmake
    NAMESPACE CorePP::
    DESTINATION lib/cmake/CorePP
)

# --- Create install directory at build time ---
add_custom_target(create_install_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}
    COMMENT "Creating install directory: ${CMAKE_INSTALL_PREFIX}"
)
