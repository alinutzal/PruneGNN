#!/bin/sh -i
# setup appropriate LCG 105 release via cvmfs

if test -e /etc/centos-release && grep 'CentOS Linux release 7' /etc/centos-release; then
  lcg_os=centos7
else 
  if test -e /etc/redhat-release && grep 'Red Hat Enterprise Linux release 9.3' /etc/redhat-release; then
    lcg_os=el9
  else
    echo "Unsupported system" 1>&2
    return
  fi
fi

lcg_release=LCG_105_a
lcg_compiler=gcc11-opt
lcg_platform=x86_64-${lcg_os}-${lcg_compiler}
lcg_view=/cvmfs/sft.cern.ch/lcg/views/${lcg_release}/${lcg_platform}
echo -e setup location: $lcg_view

source ${lcg_view}/setup.sh
# extra variables required to build acts
export DD4hep_DIR=${lcg_view}
