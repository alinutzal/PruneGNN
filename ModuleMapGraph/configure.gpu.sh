#!/bin/sh -i
source setup_cvmfs_lcg.gpu.sh
export Tracking_path=$(pwd)
export PATH=$PATH:$Tracking_path/build/bin
#export CUDA_HOME=/usr/local/cuda-12
echo -e 'installation path:' $Tracking_path
mkdir $Tracking_path/build
export PATH=$PATH:$Tracking_path/build/bin
#export PATH=$PATH:/sps/l2it/collard/cmake-3.28.2/build/Bootstrap.cmk/
#export PATH=$PATH:/sps/l2it/collard/boost/boost_1_84_0/
module load Production/cmake/3.25.0
module load Programming_Languages/python/3.9.1
module load Libraries/libtorch/1.4
module load HPC_GPU_Cloud/nvidia_hpc_sdk/21.7
