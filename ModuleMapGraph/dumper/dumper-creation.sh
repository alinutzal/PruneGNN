local_dir=$(pwd)
mkdir Dumper2
cd Dumper2

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
PS1="\[\033[1;32m\]\u\[\033[0m\]\[\033[32m\][\w]\[\033[0m\]> "
lsetup git
asetup 21.9,Athena,21.9.26
git clone https://:@gitlab.cern.ch:8443/jstark/athena.git athena
##     [This command works because I have cerated my own clone of the athena git repository. One does that once and only once.]

cd athena
git remote add upstream https://:@gitlab.cern.ch:8443/atlas/athena.git
git fetch upstream
git checkout release/21.9.26
#     [Now we have a local copy of the entire Athena code, version 21.9.26]

cd "$local_dir/Dumper2/athena/InnerDetector/InDetRecEvent/InDetPrepRawData/InDetPrepRawData/"
cp /sps/l2it/stark/NewSamples/Dumper/athena-jstark/InnerDetector/InDetRecEvent/InDetPrepRawData/InDetPrepRawData/SiCluster.h .
cd ../src/
cp /sps/l2it/stark/NewSamples/Dumper/athena-jstark/InnerDetector/InDetRecEvent/InDetPrepRawData/src/SiCluster.cxx .
cd "$local_dir/Dumper2/athena"
#     [i.e. return to the « head » directory of the local copy of athena]

cd Tracking/
cp -r /sps/l2it/stark/NewSamples/Dumper/athena-jstark/Tracking/TrkDumpAlgs/ .
cd "$local_dir/Dumper2"
#     [i.e. go ABOVE the « head » directory of the local copy of athena]

cp /sps/l2it/stark/NewSamples/Dumper/package_filters.txt .
mkdir build && cd build
cmake -DATLAS_PACKAGE_FILTER_FILE=../package_filters.txt ../athena/Projects/WorkDir
#     [This configures a « sparse build » : only the packages that are listed in package_filter.txt will be recompiled, the rest is taken from the official ATLAS release.]

make
#     [This is the actual build]

source x86_64-centos7-gcc62-opt/setup.sh 
#     [activate our sparse build for execution]

cd ..
mkdir Run
cd Run
cp /sps/l2it/stark/NewSamples/Dumper/Run/run_reco.sh .
source run_reco.sh

