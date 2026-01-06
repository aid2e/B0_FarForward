#!/bin/bash

set -e

# verif global 

: "${PROJECT_DIR:?}"
: "${EVENTS_PREFIX:?}"
: "${NARRAYS:?}"
: "${NCPUS_PER_TASK:?}"
: "${SIF_PATH:?}"

# sanitizing
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=$SSL_CERT_FILE
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export GIT_SSL_CAINFO=$SSL_CERT_FILE

unset APPTAINER_BINDPATH SINGULARITY_BINDPATH APPTAINER_BIND SINGULARITY_BIND BINDPATH APPTAINERENV_APPTAINER_BINDPATH SINGULARITYENV_SINGULARITY_BINDPATH

export SLURM_CONF=/etc/slurm/slurm.conf
which srun >/dev/null || { echo "[FATAL] srun introuvable dans le conteneur"; exit 1; }
srun -N1 -n1 -c1 --exclusive --cpu-bind=none bash -lc 'echo step ok on $(hostname)'

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# sourcing 

echo "[DEBUG run.mobo.sh] Start: sourcing botorch_env."
source "${PROJECT_DIR}/botorch_env_oldeicshell/bin/activate"
echo "[DEBUG run.mobo.sh] Achieved: sourcing botorch_env."

echo "[DEBUG run.mobo.sh] Start: sourcing EIC container."
source "${PROJECT_DIR}/Old_eic-shell/custom-eic-shell"
echo "[DEBUG run.mobo.sh] Achieved: sourcing EIC container."

echo "[DEBUG run.mobo.sh] Start: sourcing EPIC models."
source "${PROJECT_DIR}/Old_eic-shell/epic/epic_install/setup.sh"
echo "[DEBUG run.mobo.sh] Achieved: sourcing EPIC models."

export DETECTOR="epic_craterlake_18x275"
export DETECTOR_CONFIG="epic_craterlake_18x275"

echo "[DEBUG run.mobo.sh] Start: sourcing EICrecon."
source "${PROJECT_DIR}/Old_eic-shell/EICrecon/EICrecon_install/bin/eicrecon-this.sh"
echo "[DEBUG run.mobo.sh] Achieved: sourcing EICrecon."

# run

python3 -u ${PROJECT_DIR}/mobo.py
