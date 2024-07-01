#!/bin/bash

#PBS -N JOB_NAME             # Job name
#PBS -o $LOG_DIR/JOB_NAME.out  # Output file
#PBS -e $LOG_DIR/JOB_NAME.err    # Error file
#PBS -l walltime=01:00:00       # Time limit hrs:min:sec

# Load modules or software if needed
module load singularity

# Change to the directory where you submitted the job
export PBS_WORK_WORKDIR="WORK_DIR"
export PBS_O_WORKDIR="OUTPUT_DIR"
cd $PBS_O_WORKDIR

# Your script or command here
echo "Starting job"
EIC_SHELL -- SCRIPTFILE