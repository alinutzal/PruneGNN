cd Dumper2
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup git
asetup 21.9,Athena,21.9.26
source build/x86_64-centos7-gcc62-opt/setup.sh 
export PATH=$PATH:/Dumper2/athena/Reconstruction/RecJobTransforms/scripts/Reco_tf.py
sh /sps/l2it/collard/dumper/Dumper2/Run/run_reco.sh
