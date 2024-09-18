# Standalone C++ code to Compute a Module Map and to build Graph from it

## Compilation

To compile the code, use the generic scripts
```console
./Tracking.exe
```
You will see this menu, choose option 1)
```console
Welcome to the Tracking program
1) compile code	      4) run mpi version    7) get c++ standard
2) run g++ version    5) run nvcc version   8) Quit
3) run intel version  6) profile code
please choose what you want to do:
```
Then choose the compiler, recommended are `g++` (option 1) for small statistics test or `mpic++` (option 3) for runs on larger statistics (to be able to use multithreading)
```console
1) g++			3) mpic++		5) Clean
2) intel icpc	4) nvcc			6) Return to main menu
please choose a compiler:
```

## Build a Module Map

### With g++ compiler 

[Instructions to be written]

### With MPI compiler and slurm

If from a new session, first configure the package
```console
source configure.sh
````
We assume the input `.csv` files (hits and truth particles) are stored in different batches in subfolders inside a main directory. For instance:
 ```
 MY_DATA_DIR/feature_store_0/event000000001-hits.csv
 MY_DATA_DIR/feature_store_0/event000000001-particles.csv
 MY_DATA_DIR/feature_store_0/event000000002-hits.csv
 MY_DATA_DIR/feature_store_0/event000000002-particles.csv
[...]
MY_DATA_DIR/feature_store_n/event0000000xx-hits.csv
MY_DATA_DIR/feature_store_n/event0000000xx-particles.csv
```
Make a text file listing all the batch subfolders you want to run on. For instance `batch_list.txt` containing:
```txt
feature_store_0
feature_store_1
[...]
feature_store_n
```

Then launch `n` slurm jobs on each batch
```console
mkdir run
cd run
../MPI/scripts/ModuleMapScheduler.sh MY_DATA_DIR batch_list.txt my_output_dir
```
The output directory `my_output_dir` will by automatically created in the `run` directory, and the output of the jobs saved there.
Once the jobs are complete, you need to merge the `n` small Module Maps into a big one:
```console
cd run
cd my_output_dir
mv feature_store_*/*.root . # to have the small Module Map files in the same directory to merge them
cd ..
../build/mpi/bin/ModuleMapFileMerger.exe --input-dir=my_output_dir --output-module-map=my_big_module_map_name
```



| :warning:  Below the instructions are most probably not up-to date   |
|----------------------------------------------------------------------|

## Parameters for the executables are located in the following locations

## To convert csv file to root format

* define file parameters in `scripts/root_converter.sh`:
```  
    --input-dir= (containing csv files)  
    --output-dir=
```

## Events location  

* Place the event files in the directory "events"  
events should be in one of the following formats  
```
event000000001-hits.root  
event000000001-hits.csv  
event000000001-truth.csv  
event000000001-particles.root  
event000000001-particle.csv
```  
Each event should include 2 files, one called hits or truth in root or csv format, and another one called particles in root or csv format

## To build the Module Map

* define parametera the file `scripts/ModuleMapCreator.sh`
* to use slurm without mpi configure parameters in files 'MPI/scripts/ModuleMapScheduler.sh' and 'slurm/ModuleMapCreator.scheduler.slurm'
* for parallel computing define parameter in `MPI/scripts/ModuleMapCreator.mpi.sh` (giving the number of cores on which you are running the jobs to mpirun)
* to use slurm define parameters in files 'MPI/scripts/ModuleMapScheduler.sh'and in 'slurm/ModuleMapScheduler.slurm'
* to re-schedule missed events during computation define parameters in 'MPI/scripts/ModuleMapScheduler.sh' and in 'slurm/ModuleMapScheduler.slurm'

