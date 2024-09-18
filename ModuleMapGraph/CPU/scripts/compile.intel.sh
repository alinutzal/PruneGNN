cmake -H. -B build/intel -DCMAKE_CXX_COMPILER=icpc -DCMAKE_PREFIX_PATH="${PY_PATHS}"
cmake --build build/intel -- -j 4

