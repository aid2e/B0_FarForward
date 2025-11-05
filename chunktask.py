""" Simulation launcher chunked on CPUs """

# librairies 

import os
from pathlib import Path
import simu
from concurrent.futures import ProcessPoolExecutor, as_completed

# helper

def get_job_index() -> int:
    return int(
        os.getenv("JOB_INDEX")
        or os.getenv("SLURM_ARRAY_TASK_ID")
        or os.getenv("SLURM_PROCID")
        or "0"
    )

# bayesian trial

x = [float(os.environ['Z1']),
     float(os.environ['DZ2']),
     float(os.environ['DZ3']),
     float(os.environ['DZ4'])]

# path

if os.getenv("ITER_DIR"):
    it_dir = Path(os.environ["ITER_DIR"])
else:
    it_dir = simu._iter_dir(x)

i = get_job_index()
NCPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
events_prefix = os.environ['EVENTS_PREFIX']
job_dir = it_dir / f"job{i}"
job_dir.mkdir(parents=True, exist_ok=True)
input_file = job_dir / f"{events_prefix}_job{i}.edm4eic.hepevt"

# chunk by cpu

_ = simu.cpu_chunking(input_file, job_dir, NCPUS)

# launch parallel

compact = Path(os.environ['DETECTOR_PATH']) / f"{os.environ['DETECTOR_CONFIG']}.xml"
with ProcessPoolExecutor(max_workers=NCPUS) as exe:
    fut2cpu = {exe.submit(simu.run_npsim, i, j, compact, x): j for j in range(NCPUS)}
    for fut in as_completed(fut2cpu):
        j = fut2cpu[fut]
        _ = fut.result()
        print(f"[DEBUG chunktask] JOB {i}, CPU {j} DONE.")
