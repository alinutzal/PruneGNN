message (STATUS "Compiling CUDA libraries")

enable_language(CUDA)

add_library (
    GPU SHARED
    include/graph_creator.cuh
    src/graph_creator.cu
    headers/CUDA_graph_creator
)

# TODO change infrastructure such that we can install in subdirectories of include
target_include_directories(GPU PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/headers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(GPU CPU_deps Boost::python Boost::numpy Python::Python Python::NumPy)

target_compile_options (GPU INTERFACE -O3)
target_compile_features (GPU INTERFACE cxx_std_17)

# target_compile_options (GraphCreator INTERFACE -std=c++17 -lstdc++fs -O3 -allow-unsupported-compiler -lCore -lTree -lRIO -Xlinker -MMD --cudart shared)
# target_compile_features (GraphCreator INTERFACE cxx_std_17)
# target_compile_options (GraphCreator INTERFACE -O3 -D_GLIBCXX_USE_CXX11_ABI=1 -ltorch_cpu -lc10)

add_executable (GraphBuilder.cu.exe utils/GraphBuilder.cu)

target_link_libraries (GraphBuilder.cu.exe Hits GraphCreator GPU)

### Install
install(
  TARGETS GPU GraphBuilder.cu.exe
  EXPORT ModuleMapGraph
  COMPONENT GPU
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  DIRECTORY include/
  COMPONENT GPU
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
  DIRECTORY headers/
  COMPONENT GPU
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  DIRECTORY src/
  COMPONENT GPU
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
