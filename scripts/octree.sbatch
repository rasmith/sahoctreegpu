#!/bin/bash
#SBATCH -J octree           # job name
#SBATCH -o octree.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p SBATCH_QUEUE     # queue (partition) -- normal, development, etc.
#SBATCH -t 01:30:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=SBATCH_MAIL
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH -A GraviT
#./bin/octree -o ../obj/cow.obj -c cpu -b host
#./bin/octree ../obj/cow.obj -r cpu_rtp_simple
#./bin/octree ../obj/cow.obj -r gpu_rtp_simple
./bin/octree ../obj/cow.obj -r gpu_cuda_simple
