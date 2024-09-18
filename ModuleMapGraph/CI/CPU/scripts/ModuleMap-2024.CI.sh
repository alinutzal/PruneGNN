#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2024 COLLARD Christophe
#* copyright © 2024 Centre National de la Recherche Scientifique
#* copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

#exit on error
#set -e

# 2024 dataset
export year=2024

RED='\033[1;31m'
GREEN='\033[1;32m'
NC='\033[0m' # No Color

# test on a first dataset of 10 events
source $Tracking_path/CI/CPU/scripts/ModuleMapCreator.10evts.1.sh 1>/dev/null
# MM doublet 10 events
error_status=$(diff -q $Tracking_path/log/ModuleMap.$year.10evts.1.doublets.txt $Tracking_path/CI/CPU/data.$year/MM.$year.10events/ModuleMap.$year.10evts.1.doublets.txt)
echo -n "Module map doublet for 10 events:"
[[ $error_status == "" ]] && echo -e "${GREEN} OK ${NC}" || echo -e "${RED} Failed ${NC}"
# MM triplet 10 other events
error_status=$(diff -q $Tracking_path/log/ModuleMap.$year.10evts.1.triplets.txt $Tracking_path/CI/CPU/data.$year/MM.$year.10events/ModuleMap.$year.10evts.1.triplets.txt)
echo -n "Module map triplet for 10 events:"
[[ $error_status == "" ]] && echo -e "${GREEN} OK ${NC}" || echo -e "${RED} Failed ${NC}"

# test on a second dataset of 10 events
source $Tracking_path/CI/CPU/scripts/ModuleMapCreator.10evts.2.sh 1>/dev/null
# MM doublet 10 events
error_status=$(diff -q $Tracking_path/log/ModuleMap.$year.10evts.2.doublets.txt $Tracking_path/CI/CPU/data.$year/MM.$year.10events/ModuleMap.$year.10evts.2.doublets.txt)
echo -n "Module map doublet for 10 events:"
[[ $error_status == "" ]] && echo -e "${GREEN} OK ${NC}" || echo -e "${RED} Failed ${NC}"
# MM triplet 10 other events
error_status=$(diff -q $Tracking_path/log/ModuleMap.$year.10evts.2.triplets.txt $Tracking_path/CI/CPU/data.$year/MM.$year.10events/ModuleMap.$year.10evts.2.triplets.txt)
echo -n "Module map triplet for 10 events:"
[[ $error_status == "" ]] && echo -e "${GREEN} OK ${NC}" || echo -e "${RED} Failed ${NC}"

# test MM merge
source $Tracking_path/CI/CPU/scripts/ModuleMapFileMerger.sh 1>/dev/null
# MM doublet 20 events
error_status=$(diff -q $Tracking_path/log/ModuleMap.$year.20evts.doublets.txt $Tracking_path/CI/CPU/data.$year/MM.$year.20events/ModuleMap.$year.20evts.doublets.txt)
echo -n "Module Map File Merger - doublets:"
[[ $error_status == "" ]] && echo -e "${GREEN} OK ${NC}" || echo -e "${RED} Failed ${NC}"
# MM doublet 20 events
error_status=$(diff -q $Tracking_path/log/ModuleMap.$year.20evts.triplets.txt $Tracking_path/CI/CPU/data.$year/MM.$year.20events/ModuleMap.$year.20evts.triplets.txt)
echo -n "Module Map File Merger - triplets:"
[[ $error_status == "" ]] && echo -e "${GREEN} OK ${NC}" || echo -e "${RED} Failed ${NC}"


#result of test
#exit_code=$?
#if [[ $exit_status -eq 0 ]]; then
#    echo -e "${GREEN}Tests completed successfully ${NC}"
#else
#    echo -e "${RED}Tests failed ${NC}"
#fi
