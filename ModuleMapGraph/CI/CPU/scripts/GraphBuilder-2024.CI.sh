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

# CTD2023 dataset
export year=2024

# test on a first dataset of 10 events
source $Tracking_path/CI/CPU/scripts/GraphBuilder.10evts.1.sh # 1>/dev/null

# graph 10 events
for i in {1..9}
do
error_status=$(diff -q $Tracking_path/log/event00000000${i}_INPUT.txt $Tracking_path/CI/CPU/data.$year/graphs$year-1/event00000000${i}_INPUT.txt)
echo -n "graph event #$i:"
[[ $error_status == "" ]] && echo -e "\033[1;32m OK \033[1;0m" || echo -e "\033[1;31m Failed \033[1;m"
done
error_status=$(diff -q $Tracking_path/log/event000000010_INPUT.txt $Tracking_path/CI/CPU/data.$year/graphs$year-1/event000000010_INPUT.txt)
echo -n "graph event #10:"
[[ $error_status == "" ]] && echo -e "\033[1;32m OK \033[1;0m" || echo -e "\033[1;31m Failed \033[1;m"

# graph true 10 events
for i in {1..9}
do
error_status=$(diff -q $Tracking_path/log/event00000000${i}_TARGET.txt $Tracking_path/CI/CPU/data.$year/graphs$year-1/event00000000${i}_TARGET.txt)
echo -n "graph true event #$i:"
[[ $error_status == "" ]] && echo -e "\033[1;32m OK \033[1;0m" || echo -e "\033[1;31m Failed \033[1;m"
done
error_status=$(diff -q $Tracking_path/log/event000000010_TARGET.txt $Tracking_path/CI/CPU/data.$year/graphs$year-1/event000000010_TARGET.txt)
echo -n "graph true event #10:"
[[ $error_status == "" ]] && echo -e "\033[1;32m OK \033[1;0m" || echo -e "\033[1;31m Failed \033[1;m"

#source $Tracking_path/CI/CPU/scripts/GraphBuilder.10evts.2.sh 1>/dev/null

# graph 10 events
for i in {11..20}
do
error_status=$(diff -q $Tracking_path/log/event0000000${i}_INPUT.txt $Tracking_path/CI/CPU/data.$year/graphs$year-2/event0000000${i}_INPUT.txt)
echo -n "graph event #$i:"
[[ $error_status == "" ]] && echo -e "\033[1;32m OK \033[1;0m" || echo -e "\033[1;31m Failed \033[1;m"
done
error_status=$(diff -q $Tracking_path/log/event000000020_INPUT.txt $Tracking_path/CI/CPU/data.$year/graphs$year-2/event000000020_INPUT.txt)
echo -n "graph event #10:"
[[ $error_status == "" ]] && echo -e "\033[1;32m OK \033[1;0m" || echo -e "\033[1;31m Failed \033[1;m"

# graph true 10 events
for i in {11..20}
do
error_status=$(diff -q $Tracking_path/log/event0000000${i}_TARGET.txt $Tracking_path/CI/CPU/data.$year/graphs$year-2/event0000000${i}_TARGET.txt)
echo -n "graph true event #$i:"
[[ $error_status == "" ]] && echo -e "\033[1;32m OK \033[1;0m" || echo -e "\033[1;31m Failed \033[1;m"
done
error_status=$(diff -q $Tracking_path/log/event000000020_TARGET.txt $Tracking_path/CI/CPU/data.$year/graphs$year-2/event000000020_TARGET.txt)
echo -n "graph true event #10:"
[[ $error_status == "" ]] && echo -e "\033[1;32m OK \033[1;0m" || echo -e "\033[1;31m Failed \033[1;m"

