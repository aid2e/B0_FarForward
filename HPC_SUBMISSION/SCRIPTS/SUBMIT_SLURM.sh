#!/bin/bash

#SBATCH --partition=production          # Partition (queue)
#SBATCH --account=eic
#SBATCH --job-name=JOB_NAME             # Job name
#SBATCH --output=LOG_DIR/JOB_NAME.out   # Output file
#SBATCH --error=LOG_DIR/JOB_NAME.err    # Error file
#SBATCH --time=03:00:00                 # Time limit hrs:min:sec
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem-per-cpu=2G                # Main memory in MByte per CPU
#SBATCH --constraint=farm19|farm23      # node types requirement
#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ksuresh@wm.edu      # Email address


echo "RUNNING SIMULATIONS IN ${HOSTNAME}"

which singularity

if [ $? -ne 0 ]; then
    echo "Singularity not found. Exiting..."
    exit 1
fi


echo "Successfully loaded singularity $(which singularity)"

# Change to the directory where you submitted the job
cd OUTPUT_DIR

# Your script or command here
echo "Starting job"
chmod 777 SCRIPTFILE
EIC_SHELL -- SCRIPTFILE
