#!/bin/bash

#SBATCH --job-name=JOB_NAME             # Job name
#SBATCH --output=LOG_DIR/JOB_NAME.out   # Output file
#SBATCH --error=LOG_DIR/JOB_NAME.err    # Error file
#SBATCH --time=03:00:00                 # Time limit hrs:min:sec
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ksuresh@wm.edu      # Email address
#SBATCH --constraint="[gust|gulf]"


echo "RUNNING SIMULATIONS IN ${HOSTNAME}"

# Load modules or software if needed
module try-load singularity
# Check if any error occurred during the previous command
if [ $? -ne 0 ]; then
    echo "An error occurred. Exiting..."
    exit 1
fi

echo "Successfully loaded singularity $(which singularity)"

# Change to the directory where you submitted the job
cd OUTPUT_DIR

# Your script or command here
echo "Starting job"
chmod 777 SCRIPTFILE
EIC_SHELL -- SCRIPTFILE
