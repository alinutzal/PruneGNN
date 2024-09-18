# cmake -H. -Bbuild -DCMAKE_INCLUDE_PATH="/sps/l2it/collard/python.env/lib64/python3.9/site-packages/torch/include/torch/" # -DCMAKE_CXX_COMPILER=g++
cmake -H. -B build/g++ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_PREFIX_PATH="${PY_PATHS}"
cmake --build build/g++ -- -j 4
