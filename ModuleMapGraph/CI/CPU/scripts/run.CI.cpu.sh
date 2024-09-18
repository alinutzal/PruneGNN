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
ln -sf $Tracking_path/build/g++/CPU $Tracking_path/build/bin

# Creat log directory, or delete data inside
if [[ ! -d "$Tracking_path/log" ]]; then
    mkdir "$Tracking_path/log"
fi

rm $Tracking_path/log/* || true
echo -e
echo  -e "\033[33;34mPlease wait, Module Map tests in progress on the CTD2023 dataset ...\033[0m"
source $Tracking_path/CI/CPU/scripts/ModuleMap-2023.CI.sh

rm $Tracking_path/log/* || true
echo  -e "\033[33;34mPlease wait, Module Map tests in progress on the 2024 dataset ...\033[0m"
source $Tracking_path/CI/CPU/scripts/ModuleMap-2024.CI.sh

rm $Tracking_path/log/* || true
echo  -e "\033[33;34mPlease wait, Graph Builder tests in progress on the CTD2023 dataset ...\033[0m"
source $Tracking_path/CI/CPU/scripts/GraphBuilder-2023.CI.sh

#genreate dataset and adjust test (non continuouis data filenames)
#rm $Tracking_path/log/* || true
#echo  -e "\033[33;34mPlease wait, Graph Builder tests in progress on the 2024 dataset ...\033[0m"
#source $Tracking_path/CI/CPU/scripts/GraphBuilder-2024.CI.sh
