cmake_minimum_required(VERSION 3.15)
project(moonlight-python-native LANGUAGES C CXX)

include(CheckCXXSourceCompiles)

# Build moonlight-common-c as a shared library
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(ext/moonlight-common-c)

# Install the shared library so scikit-build-core can package it
install(TARGETS moonlight-common-c
    LIBRARY DESTINATION moonlight_python
    RUNTIME DESTINATION moonlight_python
)
