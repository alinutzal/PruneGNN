#!/bin/sh -i
source setup_cvmfs_lcg.cpu.sh
export Tracking_path=$(pwd)
export PATH=$PATH:$Tracking_path/build/bin
#export CUDA_HOME=/usr/local/cuda-12
echo -e 'installation path:' $Tracking_path
#export PATH=$PATH:/sps/l2it/collard/cmake-3.28.2/build/Bootstrap.cmk/
#export PATH=$PATH:/sps/l2it/collard/boost/boost_1_84_0/
#module load Compilers/intel/2020
#module load Libraries/libtorch/1.4
