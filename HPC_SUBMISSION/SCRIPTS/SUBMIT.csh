#!/bin/tcsh

#PBS -N JOB_NAME             # Job name
#PBS -o LOG_DIR/JOB_NAME.out  # Output file
#PBS -e LOG_DIR/JOB_NAME.err    # Error file
#PBS -l walltime=03:00:00       # Time limit hrs:min:sec
#PBS -l nodes=1:bora:ppn=1           # Number of nodes and cores per node

echo "RUNNING SIMULATIONS IN ${HOSTNAME}"

# Load modules or software if needed
module load singularity

# Change to the directory where you submitted the job
setenv PBS_WORK_WORKDIR "WORK_DIR"
setenv PBS_O_WORKDIR "OUTPUT_DIR"
cd $PBS_O_WORKDIR

# Your script or command here
echo "Starting job"
chmod 777 SCRIPTFILE
EIC_SHELL -- SCRIPTFILE