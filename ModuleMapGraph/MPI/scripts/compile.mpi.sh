cmake -H. -B build/mpi -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_PREFIX_PATH="${PY_PATHS}" -DUSE_MPI=ON
cmake --build build/mpi -- -j 4

