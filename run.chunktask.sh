#!/bin/bash

# verif global

: "${NCHUNKS:?}"
: "${SLURM_CPUS_PER_TASK:?}"
: "${PROJECT_DIR:?}"

# sourcing

source "${PROJECT_DIR}/botorch_env_oldeicshell/bin/activate"
source "${PROJECT_DIR}/Old_eic-shell/epic/epic_install/setup.sh"
source "${PROJECT_DIR}/Old_eic-shell/EICrecon/EICrecon_install/bin/eicrecon-this.sh"

# jobs launching

JOB_INDEX="${JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}}"
export JOB_INDEX
: "${ITER_DIR:?ITER_DIR must be set}"
ARRAY_DIR="${ITER_DIR}/arrays/array${JOB_INDEX}"
export ARRAY_DIR
export SLURM_ARRAY_TASK_ID="${SLURM_PROCID}"

"$VIRTUAL_ENV/bin/python3" -u "${PROJECT_DIR}/chunktask.py"
