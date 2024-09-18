find_package(Torch REQUIRED)

# Set this up to propagate options to libraries
add_library(CPU_deps INTERFACE)

# TODO change infrastructure such that we can install in subdirectories of include
target_include_directories(CPU_deps INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/headers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_compile_options (CPU_deps INTERFACE -O3)
target_compile_features (CPU_deps INTERFACE cxx_std_17)

target_link_libraries(CPU_deps INTERFACE
    ROOT::Core ROOT::Tree
    Boost::program_options
    ${TORCH_LIBRARIES}
)

add_library(
GraphCreator SHARED
include/colors.hpp
include/common_options.hpp
src/common_options.cpp
headers/colors
headers/common_options
headers/parameters
)

target_link_libraries(GraphCreator PUBLIC CPU_deps)
target_compile_options(GraphCreator PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Werror>)
target_compile_options (GraphCreator INTERFACE -O3 -D_GLIBCXX_USE_CXX11_ABI=1 -ltorch_cpu -lc10)

# ============
# Hits Library
# ============

add_library(
Hits SHARED
include/TTree_hits.hpp
include/hit.hpp
include/hits.hpp
include/hits_infos.hpp
src/TTree_hits.cpp
src/hit.cpp
src/hits.cpp
src/hits_infos.cpp
headers/TTree_hits
headers/hit
headers/hits
headers/hits_infos
)

target_link_libraries(Hits PUBLIC CPU_deps)

# =================
# Particles Library
# =================

add_library(
Particles SHARED
include/TTree_particles.hpp
include/particle.hpp
include/particles.hpp
src/TTree_particles.cpp
src/particle.cpp
src/particles.cpp
headers/TTree_particles
headers/particle
headers/particles
)

target_link_libraries(Particles PUBLIC CPU_deps)

# ==================
# Module Map Library
# ==================

add_library(
ModuleMap SHARED
include/module_doublet.hpp
include/module_triplet.hpp
include/module_map_triplet.hpp
src/module_doublet.cpp
src/module_triplet.cpp
src/module_map_triplet.cpp
headers/module_doublet
headers/module_triplet
headers/module_map_triplet
)

target_link_libraries(ModuleMap PUBLIC CPU_deps)

# =============
# Graph Library
# =============

add_library(
Graph SHARED
include/edge.hpp
include/edge_true.hpp
include/vertex.hpp
include/vertex_true.hpp
include/graph.hpp
include/graph_true.hpp
include/graph_creator.hpp
include/geometric_cuts.hpp
include/strip_hit_pair.hpp
include/strip_module_DB.hpp
include/event_id.hpp
src/edge.cpp
src/edge_true.cpp
src/vertex.cpp
src/graph.cpp
src/graph_true.cpp
src/graph_creator.cpp
src/geometric_cuts.cpp
src/strip_hit_pair.cpp
src/strip_module_DB.cpp
src/event_id.cpp
headers/edge
headers/edge_true
headers/vertex
headers/vertex_true
headers/graph
headers/graph_true
headers/graph_creator
headers/geometric_cuts
headers/strip_hit_pair
headers/strip_module_DB
)

target_link_libraries(Graph PUBLIC CPU_deps Boost::python Boost::numpy Python::Python Python::NumPy)

# This library allows conveiently adding all the sublibraries as one target
# TODO maybe refactor them into one actual shared library?
add_library(CPU INTERFACE)
target_link_libraries(CPU INTERFACE Hits Particles ModuleMap Graph)


# ===============
# CPU Executables
# ===============
add_executable (c++.vs.exe utils/c++.vs.cpp)
add_executable (ModuleMapCreator.exe utils/ModuleMapCreator.cpp)
add_executable (GraphBuilder.exe utils/GraphBuilder.cpp)
add_executable (converter.exe utils/root_converter.cpp)
add_executable (ModuleMapFileMerger.exe utils/ModuleMapFileMerger.cpp)
add_executable (ModuleMapCyclesCleaner.exe utils/ModuleMapCyclesCleaner.cpp)
add_executable (ModuleMapRoot2Txt.exe utils/ModuleMapRoot2Txt.cpp)

target_link_libraries (c++.vs.exe)
target_link_libraries (ModuleMapCreator.exe Hits Particles GraphCreator ModuleMap )
target_link_libraries (GraphBuilder.exe Hits Particles GraphCreator Graph )
target_link_libraries (converter.exe Hits Particles GraphCreator )
target_link_libraries (ModuleMapFileMerger.exe CPU_deps)
target_link_libraries (ModuleMapCyclesCleaner.exe CPU_deps)
target_link_libraries (ModuleMapRoot2Txt.exe CPU_deps)

# =======
# Install
# =======

install(
  TARGETS
    CPU CPU_deps GraphCreator Hits Graph Particles ModuleMap
    c++.vs.exe ModuleMapCreator.exe GraphBuilder.exe converter.exe
    ModuleMapFileMerger.exe ModuleMapCyclesCleaner.exe ModuleMapRoot2Txt.exe
  EXPORT ModuleMapGraph
  COMPONENT CPU
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
  DIRECTORY include/
  COMPONENT CPU
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  DIRECTORY headers/
  COMPONENT CPU
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  DIRECTORY src/
  COMPONENT CPU
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
