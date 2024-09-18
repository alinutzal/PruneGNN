#cmake -H. -B build/nvc++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12 -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_PREFIX_PATH="${PY_PATHS}" # -DCMAKE_CUDA_ARCHITECTURES=native
cmake -H. -B build/nvc++ -DCMAKE_CXX_STANDARD=17  -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" -DCMAKE_CXX_FLAGS=-std=c++17 -DCMAKE_PREFIX_PATH="${PY_PATHS}"
cmake --build build/nvc++ --
