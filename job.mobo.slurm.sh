#!/bin/bash
#SBATCH --gres=disk:4G
#SBATCH --account=eic
#SBATCH --partition=production
#SBATCH --job-name=BOB0
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%A_%t.chunk.out
#SBATCH --error=logs/%A_%t.chunk.err

set -e

# sourcing user variables env

if [ -f "bobo.env" ]; then
  source "bobo.env"
else
  echo "[FATAL] File bobo.env not found."
  exit 1
fi
: "${PROJECT_DIR:?PROJECT_DIR must be defined in bobo.env}"
: "${NARRAYS:?NARRAYS must be defined in bobo.env}"
: "${NCPUS_PER_TASK:?NCPUS_PER_TASK must be defined in bobo.env}"
: "${LOWER_BOUNDS:?LOWER_BOUNDS must be defined in bobo.env}"
: "${UPPER_BOUNDS:?UPPER_BOUNDS must be defined in bobo.env}"
: "${N_INIT:?N_INIT must be defined in bobo.env}"
: "${N_ITER:?N_ITER must be defined in bobo.env}"
: "${OBS_OBJ:?OBS_OBJ must be defined in bobo.env}"
: "${EVENTS_PREFIX:?EVENTS_PREFIX must be defined in bobo.env}"

# fixed global parameters

export TIME_BUDGET_SEC=$((79000))

# eic-shell container copy

echo "[DEBUG job.mobo.slurm.sh] Start: copy container."
SIF_SRC="${PROJECT_DIR}/Old_eic-shell/eic-jug_xl-nightly-2024-03-12.sif"
SIF_BASENAME="$(basename "$SIF_SRC")"
SCR_BASE="/scratch/slurm/${SLURM_JOB_ID}"
SIF_LOCAL="${SCR_BASE}/${SIF_BASENAME}"
srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
     --export=ALL,SIF_SRC="${SIF_SRC}",SIF_BASENAME="${SIF_BASENAME}",SCR_BASE="${SCR_BASE}" \
     bash -lc '
  set -euo pipefail
  SCR="$SCR_BASE"
  mkdir -p "$SCR/cache" "$SCR/tmp"
  dst="$SCR/$SIF_BASENAME"
  if [ ! -s "$dst" ]; then
    cp -f "$SIF_SRC" "$dst"
  fi
'
echo "[DEBUG job.mobo.slurm.sh] Achieved: copy container." 

export APPTAINER_CACHEDIR="${SCR_BASE}/cache"
export APPTAINER_TMPDIR="${SCR_BASE}/tmp"

# run

echo "[DEBUG job.mobo.slurm.sh] Start: running on $(hostname)."

cat > "${SCR_BASE}/singularity-shim" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "${SCR_BASE}/singularity-shim"
export SIF_PATH="$SIF_LOCAL"

unset APPTAINER_BINDPATH SINGULARITY_BINDPATH APPTAINER_BIND SINGULARITY_BIND BINDPATH

env -u APPTAINER_BINDPATH -u SINGULARITY_BINDPATH -u APPTAINER_BIND -u SINGULARITY_BIND -u BINDPATH \
apptainer exec \
  --env-file bobo.env \
  --overlay ${PROJECT_DIR}/slurm-overlay.img \
  -B ${PROJECT_DIR}:${PROJECT_DIR} \
  -B /usr/bin:/hostbin \
  -B ${SCR_BASE}/singularity-shim:/usr/bin/singularity \
  -B /usr/lib64:/hostlib64 \
  -B /usr/lib64/slurm:/usr/lib64/slurm \
  -B /run/munge:/run/munge -B /var/run/munge:/var/run/munge \
  "$SIF_LOCAL" ${PROJECT_DIR}/run.mobo.sh

echo "[DEBUG job.mobo.slurm.sh] Start: running on $(hostname)."