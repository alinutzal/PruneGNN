message (STATUS "Compiling MPI libraries")

find_package(MPI REQUIRED)

add_library (
    MPI SHARED
    src/send.cpp
    src/receive.cpp
    headers/send
    headers/receive
)

target_link_libraries(MPI PUBLIC CPU_deps MPI::MPI_CXX)

# TODO change infrastructure such that we can install in subdirectories of include
target_include_directories(MPI PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/headers>
    $<INSTALL_INTERFACE:>
)
target_include_directories(MPI PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)


add_executable (ModuleMapCreator.mpi.exe utils/ModuleMapCreator.mpi.cpp)
add_executable (ModuleMapScheduler.exe utils/ModuleMapScheduler.cpp)
add_executable (ModuleMapReschedule.exe utils/ModuleMapReschedule.cpp)


target_link_libraries (ModuleMapCreator.mpi.exe Hits Particles GraphCreator ModuleMap MPI CPU_deps MPI::MPI_CXX)
target_link_libraries (ModuleMapScheduler.exe MPI::MPI_CXX CPU_deps)
target_link_libraries (ModuleMapReschedule.exe MPI::MPI_CXX CPU_deps)


### Install
install(
  TARGETS MPI ModuleMapCreator.mpi.exe ModuleMapScheduler.exe ModuleMapReschedule.exe
  EXPORT ModuleMapGraph
  COMPONENT MPI
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
