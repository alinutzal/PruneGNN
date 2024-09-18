#!/usr/bin/bash

csvDir=/sps/atlas/a/avallier/itk/GNN4ITkTeam/data/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/feature_store_ttbar_uncorr
exeDir=/sps/atlas/a/avallier/itk/GNN4ITkTeam//ModuleMapGraph/build/bin
workDir=$PWD

echo "workdir=${workDir}"

for sample in train val test; do
    
    nfperjob=40

    i=1
    j=1

    for f in $csvDir/${sample}set/event*-particles.csv; do
        base_particle=${f##*/}
        full_hits=${f//particles/truth}
        base_hits=${base_particle//particles/truth}

        job_dir=jobs_$sample/job_$j

        if [ ! -d $job_dir ]; then
            mkdir -p $job_dir/graphs
            mkdir $job_dir/csv
        fi

        if (( $i % $nfperjob == 0)); then
            echo $i
            
            ln -s $f $job_dir/csv/$base_particle
            ln -s $full_hits $job_dir/csv/$base_hits
            ln -s $exeDir/GraphBuilder.exe $job_dir/GraphBuilder.exe
            cp makeGraphsSlurm.sh $job_dir/makeGraphsSlurm.sh
            cp ../setup_cvmfs_lcg103.sh $job_dir/setup_cvmfs_lcg103.sh

            cd $job_dir
            sbatch makeGraphsSlurm.sh ./csv/ ./graphs/
            #echo "sbatch makeGraphsSlurm.sh ./csv/ ./graphs/"
            cd $workDir
            
            j=$((j+1))
        else
            ln -s $f $job_dir/csv/$base_particle
            ln -s $full_hits $job_dir/csv/$base_hits
        fi  

        i=$((i+1))

    done

    echo $sample $i

done