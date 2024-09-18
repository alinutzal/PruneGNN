#!/bin/sh -i
# install software on computer
# cmake 3.28.2
# Python
# numpy
# boost 1.84
# root

#source setup_cvmfs_lcg.sh
export Tracking_path=$(pwd)
echo -e 'installation path:' $Tracking_path
export PATH=$PATH:$Tracking_path/build/bin
#export PATH=$PATH:/sps/l2it/collard/cmake-3.28.2/build/Bootstrap.cmk/
#export PATH=$PATH:/sps/l2it/collard/boost/boost_1_84_0
# The following directo
#export LD_LIBRARY_PATH=/sps/l2it/collard/boost/boost_1_84_0/stage/lib:$LD_LIBRARY_PATH
#module load Compilers/intel/2020
#module load Libraries/libtorch/1.4
ccenv boost 1.84.0
export PATH=$PATH:/pbs/software/centos-8-x86_64/boost/1.84.0/lib/cmake/Boost-1.84.0:/pbs/software/centos-8-x86_64/root/6.28.12:/pbs/software/centos-8-x86_64/root/6.28.12/share/root/cmake:/pbs/software/centos-8-x86_64/root/6.28.12/share/root/cmake/modules:/pbs/software/centos-8-x86_64/root/6.28.12/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pbs/software/centos-8-x86_64/root/6.28.12/lib/:/pbs/software/centos-8-x86_64/boost/1.84.0/lib
export VDT_INCLUDE_DIR=/pbs/software/centos-8-x86_64/root/6.28.12/include/root
