cmake_minimum_required(VERSION 3.25)

# Project's name
project (GraphCreator
         VERSION 0.2
         LANGUAGES CXX)

cmake_policy(SET CMP0115 NEW)

# Install project in standard bin/include/lib directories
include(GNUInstallDirs)

# Setup compile options
option(USE_MPI OFF "Build MPI version")
option(USE_CUDA OFF "Build CUDA version")

# minimal dependency versions. they are defined here in a single place so
# they can be easily upgraded, although they might not be used if the
# dependency is included via `add_subdirectory(...)`.
set (boost_version 1.82)
set (root_version 6.28)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard used for compiling")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


#####################
# Required packages #
#####################

# NOTE we use boost::filesystem instead of std::filesystem since the later
# is not uniformly supported even on compilers that nominally support C++17
find_package (Boost ${boost_version} REQUIRED CONFIG COMPONENTS filesystem program_options unit_test_framework graph python numpy)

find_package(ROOT ${root_version} REQUIRED CONFIG COMPONENTS Core Geom GenVector Hist Tree TreePlayer)

find_package(Threads)

find_package(Python COMPONENTS Interpreter Development NumPy)


#####################
# Setup subprojects #
#####################

add_subdirectory(CPU)

if (${USE_MPI})
  add_subdirectory(MPI)
endif()

if(${USE_CUDA})
  add_subdirectory(GPU)
endif()



# ===================
# Handle installation
# ===================

set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/ModuleMapGraph")

# This file is used in the installation process to find the dependencies
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/ModuleMapGraphConfig.cmake.in
  ${CMAKE_BINARY_DIR}/ModuleMapGraphConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
  PATH_VARS
    CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
    CMAKE_INSTALL_BINDIR CMAKE_INSTALL_CMAKEDIR
)

install(
  FILES ${CMAKE_BINARY_DIR}/ModuleMapGraphConfig.cmake
  DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
)

# Finally install the target
install(
  EXPORT ModuleMapGraph
  NAMESPACE ModuleMapGraph::
  FILE ModuleMapGraphConfig-targets.cmake
  DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
)
