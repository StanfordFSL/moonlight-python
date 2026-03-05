cmake_minimum_required(VERSION 3.15)
project(moonlight-python-native LANGUAGES C CXX)

include(CheckCXXSourceCompiles)

# Build moonlight-common-c as a shared library
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
# Export all symbols from the DLL on Windows (Linux/macOS do this by default)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_subdirectory(ext/moonlight-common-c)

# Install the shared library so scikit-build-core can package it
install(TARGETS moonlight-common-c
    LIBRARY DESTINATION moonlight_python
    RUNTIME DESTINATION moonlight_python
)
