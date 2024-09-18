#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2024 COLLARD Christophe
#* copyright © 2024 Centre National de la Recherche Scientifique
#* copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

while true; do

TO_RUN=$(whiptail --title "Compilation Menu" --menu "Choose a compiler" 25 78 10   \
"g++" "     compile codes with g++ compiler" \
"icpc" "     compile codes with intel compiler" \
"mpic++" "     compile codes with mpi compiler" \
"nvcc" "     compile codes with cuda compiler" \
"nvc++" "     compile codes with nvc++ compiler" \
"clean" "     delete compiled versions" \
"log g++" "     view g++ compilation log" \
"log intel" "     view icpc compilation log" \
"Back" "     return to main menu" 3>&1  1>&2 2>&3
)

  case $TO_RUN in
    "g++")
      source $Tracking_path/scripts/compile.g++.sh > log/g++.log 2>&1 &
      ;;
    "icpc")
      source $Tracking_path/scripts/compile.intel.sh > log/intel.log 2>&1 &
      ;;
    "mpic++")
      source $Tracking_path/MPI/scripts/compile.mpi.sh
      ;;
    "nvcc")
      source $Tracking_path/GPU/scripts/compile.nvcc.sh
      ;;
    "nvc++")
      source $Tracking_path/GPU/scripts/compile.nvc++.sh
      ;;
    "log g++")
      whiptail --title "g++ log" --textbox --scrolltext log/g++.log 50 150
      ;;
    "log intel")
      whiptail --title "intel log" --textbox --scrolltext log/intel.log 50 150
      ;;
    "Clean")
      source $Tracking_path/scripts/clean.sh
      ;;
    "Back")
      break
      ;;
    *) echo -e "\033[31minvalid option\033[0m $REPLY"
      ;;
  esac
  REPLY=
done
