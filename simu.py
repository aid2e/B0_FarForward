""" Simulation launcher functions """


# librairies


import os, subprocess, json, pathlib, time, shutil, errno
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *
import json, hashlib, textwrap
from pathlib import Path


# path managers


def _iter_tag(x):
    """
    Generator of a unique tag for the trial.

    Args:
        x (d-D torch): Geometrical parameters to optimize.
    
    Returns: 
        (str): Unique tag for the trial.
    """

    s = json.dumps([float(v) for v in x], separators=(',',':'))
    return "it_" + hashlib.sha1(s.encode()).hexdigest()[:10]


def _iter_dir(x):
    """
    Generator of a unique direction for the trial.

    Args:
        x (d-D torch): Geometrical parameters to optimize.
    
    Returns: 
        (Path): Unique direction for the trial.
    """

    return PROJECT_DIR / "bayesian_iterations" / _iter_tag(x)


def _discover_shared_sif():
    """
    Search for the SIF available for all nodes.
    
    Returns: 
        (Path): SIF dir.
    """

    s = os.environ.get("SIF_SRC")
    if s and Path(s).exists():
        return s
    cand_dir = PROJECT_DIR / "Old_eic-shell"
    cands = sorted(cand_dir.glob("*.sif"))
    if cands:
        return str(cands[0])
    raise RuntimeError("[DEBUG simu._discover_shared_sif] No SIF found.")


#  helper functions


_HOSTBIN = Path("/hostbin")


def _pick_bin(name: str) -> str:
    cand = _HOSTBIN / name
    if cand.exists():
        return str(cand)
    found = shutil.which(name)
    return found or name


SBATCH = _pick_bin("sbatch")
SQUEUE = _pick_bin("squeue")
SACCT  = _pick_bin("sacct")


def _sbatch(cmd: str, env=None) -> int:
    print("[DEBUG] sbatch cmd:", cmd)
    p = subprocess.run(cmd, env=env, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.stdout.strip():
        print("[DEBUG] sbatch out:", p.stdout.strip())
    if p.stderr.strip():
        print("[DEBUG] sbatch err:", p.stderr.strip())
    if p.returncode != 0:
        raise RuntimeError(f"sbatch failed rc={p.returncode}")
    out = p.stdout.strip()
    # --parsable -> "54880xxx" ; sinon "Submitted batch job 54880xxx"
    tok = out.split()[-1]
    return int(tok) if tok.isdigit() else int(out)


def _wait_array_done(jobid: int, poll_s: int = 10) -> str:
    while True:
        q = subprocess.run(f"{SQUEUE} -j {jobid} -h", shell=True, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if q.stdout.strip():
            time.sleep(poll_s)
            continue
        # squeue vide: on tente sacct pour l'état final (si dispo)
        st = subprocess.run(
            f"{SACCT} -j {jobid} --format=JobIDRaw,State --noheader | awk 'NR==1{{print $2}}'",
            shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        ).stdout.strip()
        return st or "UNKNOWN"


def _safe_clear_dir(dpath):
    d = Path(dpath)
    d.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        ok = True
        for p in list(d.iterdir()):
            try:
                if p.name.startswith(".nfs"):
                    # laissé en place; retentera au prochain passage
                    ok = False
                    continue
                if p.is_file() or p.is_symlink():
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError as e:
                if e.errno in (errno.ENOTEMPTY, errno.EBUSY):
                    ok = False
                # on ignore tout et on retentera
        if ok:
            return
        time.sleep(0.2 * (attempt + 1))


# reconstruction function


def run_eicrecon(input_file, cpu_dir):
    """
    Run eicrecon command.
        
    Args:
        input_file (str): NPSIM output file path. 
        cpu_dir (str): CPU path.
    
    Returns:
        (str): Output reconstruction file for the job (.root).

    """

    detector_path = os.environ.get("DETECTOR_PATH")
    detector_config = os.environ.get("DETECTOR_CONFIG")
    log_out = cpu_dir / "recon_log.out"
    log_err = cpu_dir / "recon_log.err"

    jana_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    cmd = [
        "eicrecon",
        f"-Pdd4hep:xml_file={detector_path}/{detector_config}.xml",
        "-Ppodio:output_include_collections=ReconstructedParticles,GeneratedParticles,ReconstructedChargedParticles,BoTrackerRecHits",
        f"-Pjana:nthreads={jana_threads}",
        input_file
    ]

    with open(log_out, "w") as out, open(log_err, "w") as err:
        subprocess.run(cmd, cwd=cpu_dir, stdout=out, stderr=err)

    podio_file = Path("podio_output.root").resolve()
    return(podio_file)


# run npsim function


def run_npsim(i, j, compact_file, x):
    """
    Run npsim command.
        
    Args:
        i (int): Job index.
        j (int): CPU index.
        compact_file (str): Compact file name. 
        x (list float): Geometrical parameters.
    
    Returns:
        output_file (str): Output simulation file for the job (ROOT).

    """

    # paths 

    it_dir = _iter_dir(x)
    job_dir = it_dir / f"job{i}"
    cpu_dir = job_dir / f"cpu{j}"
    steering_file = PROJECT_DIR / 'steering_new.py'
    input_file = cpu_dir / f'{EVENTS_PREFIX}_cpu{j}.edm4eic.hepevt'
    output_file = cpu_dir / f'{EVENTS_PREFIX}_{_iter_tag(x)}_job{i}_cpu{j}.root'
    log_file = cpu_dir / f'sim_log{j}.out'
    err_file = cpu_dir / f'sim_log{j}.err'

    # npsim command

    seed = 1000 + i + j

    with input_file.open("r") as f:
        events = f.readlines()
    nb_events = len(events)//2 - 1

    cmd = [
        "npsim",
        "--steeringFile", str(steering_file),
        "--inputFiles", str(input_file),
        "--random.seed", str(seed),
        "--numberOfEvents", str(nb_events),
        "--runType", "batch",
        "--compactFile", str(compact_file),
        "--outputFile", str(output_file)
        #"-v", "DEBUG", # or "DEBUG", etc. See npsim --help
    ]

    print(f"[DEBUG simu.run_npsim] Start: simulation - job {i} - cpu {j}.")

    with open(log_file, "w") as out, open(err_file, "w") as err:
        process = subprocess.run(cmd, cwd=cpu_dir, stdout=out, stderr=err)

    # end

    if process.returncode == 0:
        print(f"[DEBUG simu.run_npsim] Achieved: simulation - job {i} - cpu {j}.")
    else:
        print(f"[DEBUG simu.run_npsim] Error: simulation - job {i} - cpu {j}. Code {process.returncode}). See log & err files.")

    return output_file


# merging functions


def capped_hadd(outfile, inputs, max_gib=75):
    """
    Merging root files while total size less than a threshold.
        
    Args:
        outfile (str): Direction of the output file.
        inputs (list): List of files to merge.
        max_gib (int): Maximum size to merge (gigabites)
    
    Returns:
        outfile (str): Direction of the output file.
    """

    max_bytes = int(max_gib * (1024**3))
    sel, total = [], 0

    for p in inputs:
        try:
            s = os.path.getsize(p)
        except OSError:
            continue
        if total + s <= max_bytes:
            sel.append(p)
            total += s
        else:
            break

    if not sel:
        sel = [min(inputs, key=lambda x: os.path.getsize(x))]
        total = os.path.getsize(sel[0])

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    if len(sel) == 1:
        if os.path.abspath(sel[0]) != os.path.abspath(outfile):
            subprocess.run(["cp", "-f", sel[0], outfile], check=True)
        return outfile

    subprocess.run(["hadd", "-f", outfile, *sel], check=True)
    return outfile


def simu_merger(narrays, ncpus, x):
    """
    Merging simulation files (not reconstructed).
        
    Args:
        b (int): Bayesian iteration index.
        narrays (int): Number of arrays.
        ncpus (int): Number of cpus.
        x (list float): Test geometry parameters.
    
    Returns:
        merged_file (str): Merged file path.
    """

    it_dir = _iter_dir(x) 
    simu_dir = PROJECT_DIR / "simu"
    simu_dir.mkdir(exist_ok=True)
    files_to_merge = []

    for i in range(narrays):

        job_dir = it_dir / f"job{i}"

        for j in range(ncpus):

            cpu_dir = job_dir / f"cpu{j}"
            simu_files = list(cpu_dir.glob(f"{EVENTS_PREFIX}_{_iter_tag(x)}_job{i}_cpu{j}.root")) 

            if simu_files:
                files_to_merge.append(simu_files[0])
            else:
                print(f"[DEBUG simu.simu_merger] No simu file found in {cpu_dir} for merging.")

    if files_to_merge:

        merged_file = simu_dir / f"{EVENTS_PREFIX}_{_iter_tag(x)}_simumerge.root" 
        merged_file = capped_hadd(merged_file, files_to_merge, max_gib=85)
        print(f"[DEBUG simu.simu_merger] Merged file: {merged_file}")

    else:

        print("[DEBUG simu.simu_merger] No simu file found to merge.")
    
    return merged_file


def recon_merger(narrays, ncpus, x):
    """
    Merging reconstructed files.
        
    Args:
        narrays (int): Number of arrays.
        ncpus (int): Number of cpus.
        x (list float): Test geometry parameters.
    
    Returns:
        merged_file (str): Merged file path.
    """

    z1, dz2, dz3, dz4 = x

    simu_dir = PROJECT_DIR / "simu"
    simu_dir.mkdir(exist_ok=True)
    files_to_merge = []

    for i in range(narrays):

        job_dir = PROJECT_DIR / f"job{i}"

        for j in range(ncpus):

            cpu_dir = job_dir / f"cpu{j}"
            podio_files = list(cpu_dir.glob("podio_output.root"))

            if podio_files:
                files_to_merge.append(podio_files[0])
            else:
                print(f"[DEBUG simu.recon_merger] No podio file found in {cpu_dir} for merging.")

    if files_to_merge:

        merged_file = simu_dir / f"{EVENTS_PREFIX}_z1_{z1}_dz2_{dz2}_dz3_{dz3}_dz4_{dz4}_podiomerge.root"
        cmd = ["hadd", "-f", str(merged_file)] + [str(f) for f in files_to_merge]
        subprocess.run(cmd, check=True)
        print(f"[DEBUG simu. recon_merger] Merged file: {merged_file}")

    else:

        print("[DEBUG simu.recon_merger] No podio file found to merge.")
    
    return merged_file


# geometry modif functions


def geom_modif(DETECTOR_PATH, x):
    """
    Modifying geometry. 
        
    Args:
        DETECTOR_PATH (str): Detector path.
        x (list float): Test geometry parameters.
    
    Returns:
        b0_tracker_file (str): B0 detector path.
    """

    b0_tracker_file = Path(DETECTOR_PATH)/"compact/far_forward/B0_tracker.xml"

    z1, dz2, dz3, dz4 = x

    custom1 = {"B0TrackerLayer1_zstart":f"B0Tracker_length/2.0+{z1}"}
    custom2 = {"B0TrackerLayer2_zstart":f"B0Tracker_length/2.0+{z1}+{dz2}"}
    custom3 = {"B0TrackerLayer3_zstart":f"B0Tracker_length/2.0+{z1}+{dz2}+{dz3}"}
    custom4 = {"B0TrackerLayer4_zstart":f"B0Tracker_length/2.0+{z1}+{dz2}+{dz3}+{dz4}"}

    tree = ET.parse(b0_tracker_file)
    root = tree.getroot()
    found = set()

    for const in root.findall(".//constant"):
        name = const.get("name")
        if name in custom1:
            new_value = custom1[name]
            const.set("value", new_value)
            found.add(name)
        if name in custom2:
            new_value = custom2[name]
            const.set("value", new_value)
            found.add(name)
        if name in custom3:
            new_value = custom3[name]
            const.set("value", new_value)
            found.add(name)
        if name in custom4:
            new_value = custom4[name]
            const.set("value", new_value)
            found.add(name)

    tree.write(b0_tracker_file, encoding="utf-8", xml_declaration=True)

    return b0_tracker_file


# chunking functions


def array_chunking(input_file, dir, narrays):
    """
    Chunking inputs for each array and creating associated job folders.
        
    Args:
        input_file (str): Input file path.
        dir (str): Origin path (Bayesian iteration path).
        narrays (int): Number of arrays.
    
    Returns:
        chunk_size (int): Size of chunks per array.
    """

    with input_file.open("r") as f:
        lines = f.readlines()
    events = [lines[i:i+2] for i in range(0, len(lines), 2)]
    total_events = len(events)
    chunk_size = total_events // narrays
    print(f"[DEBUG simu.array_chunking] Arrays chunking: {narrays} arrays - {chunk_size} events per job.")

    for i in range(narrays):

        job_dir = Path(dir) / f"job{i}"
        _safe_clear_dir(job_dir)

        start = i * chunk_size
        end = (i + 1) * chunk_size if i < narrays - 1 else total_events - 1
        chunk = events[start:end]
        intput_file_job = job_dir / f'{EVENTS_PREFIX}_job{i}.edm4eic.hepevt'
        with intput_file_job.open("w") as f:
            for event in chunk:
                f.writelines(event)

    return chunk_size


# chunking cpus inside an array (independent job)


def cpu_chunking(input_file, dir, ncpus):
    """
    Chunking inputs for each cpu and creating associated job folders.
        
    Args:
        input_file (str): Input file path.
        dir (str): Origin path (task path).
        ncpus (int): Number of cpus.
    
    Returns:
        chunk_size (int): Size of chunks per cpu.
    """

    with input_file.open("r") as f:
        lines = f.readlines()
    events = [lines[i:i+2] for i in range(0, len(lines), 2)]
    total_events = len(events)
    chunk_size = total_events // ncpus
    print(f"[DEBUG simu.cpu_chunking] CPUs chunking: {ncpus} cpus - {chunk_size} events per cpu.")

    for j in range(ncpus):

        cpu_dir = Path(dir) / f"cpu{j}"
        _safe_clear_dir(cpu_dir)

        start = j * chunk_size
        end = (j + 1) * chunk_size if j < ncpus - 1 else total_events - 1
        chunk = events[start:end]
        intput_file_cpu = cpu_dir / f'{EVENTS_PREFIX}_cpu{j}.edm4eic.hepevt'
        with intput_file_cpu.open("w") as f:
            for event in chunk:
                f.writelines(event)

    return chunk_size


# simulation launching function

def launch(x):
    """
    Geant4 simulations launcher.
        
    Args:
        x (array): Bayesian trial.
    
    Returns:
        simu_merged_file (str): Path to simulation file after merging.
    """

    # parameters

    T  = int(os.environ.get("NARRAYS", "100"))
    C  = int(os.environ.get("NCPUS_PER_TASK", "8"))
    part = os.environ.get("SLURM_PARTITION", "production")
    throttle = os.environ.get("ARRAY_MAX_CONCURRENCY", "32")
    SIF = _discover_shared_sif()

    # geometry modification

    print(f"[DEBUG simu.launch] Start: geometry modification - x={x}.")
    _ = geom_modif(os.environ.get("DETECTOR_PATH"), x)
    print(f"[DEBUG simu.launch] Achieved: geometry modification - x={x}.")

    # iteration folder

    it_folder = _iter_dir(x)
    print(f"[DEBUG simu.launch] Iteration folder: {it_folder}.")
    (it_folder / "logs").mkdir(parents=True, exist_ok=True)

    # array chunking

    print(f"[DEBUG simu.launch] Start: array chunking.")
    events_dir = PROJECT_DIR / 'events/hepevt'
    input_file = events_dir / f"{os.environ['EVENTS_PREFIX']}.edm4eic.hepevt"
    _ = array_chunking(input_file, it_folder, T)
    print(f"[DEBUG simu.launch] Achieved: array chunking.")

    # array entry

    SIF_SHARED = _discover_shared_sif()
    entry_sh = it_folder / "array_entry.sh"

    entry_sh.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        set -x
        echo "[entry] host=$(hostname) jobid=${{SLURM_JOB_ID:-NA}} task=${{SLURM_ARRAY_TASK_ID:-NA}} user=$(whoami)"
        echo "[entry] PATH=$PATH"
        echo "[entry] PROJECT_DIR={PROJECT_DIR}"
        echo "[entry] SIF_SRC={SIF_SHARED}"

        if [ -f "{PROJECT_DIR}/bobo.env" ]; then
          source "{PROJECT_DIR}/bobo.env"
        else
          echo "[FATAL] File {PROJECT_DIR}/bobo.env not found."
          exit 1
        fi

        # Stage-in SIF local to THIS array job
        SCR_BASE=/scratch/slurm/${{SLURM_JOB_ID}}
        mkdir -p "${{SCR_BASE}}/cache" "${{SCR_BASE}}/tmp"
        SIF_BASENAME="$(basename "{SIF_SHARED}")"
        SIF_LOCAL="${{SCR_BASE}}/${{SIF_BASENAME}}"
        if [[ ! -s "$SIF_LOCAL" ]]; then
        echo "[entry] copying {SIF_SHARED} -> $SIF_LOCAL"
        cp -f "{SIF_SHARED}" "$SIF_LOCAL"
        else
        echo "[entry] reuse staged SIF at $SIF_LOCAL"
        fi
        export APPTAINER_CACHEDIR="${{SCR_BASE}}/cache"
        export APPTAINER_TMPDIR="${{SCR_BASE}}/tmp"

        JOB_INDEX="${{SLURM_ARRAY_TASK_ID:-${{SLURM_PROCID:-0}}}}"
        export JOB_INDEX

        apptainer exec \
        -B {PROJECT_DIR}:{PROJECT_DIR} \
        -B /usr/bin:/hostbin \
        -B /usr/lib64:/hostlib64 \
        -B /usr/lib64/slurm:/usr/lib64/slurm \
        -B /run/munge:/run/munge -B /var/run/munge:/var/run/munge \
        --env SLURM_JOB_ID="${{SLURM_JOB_ID:-}}" \
        --env SLURM_ARRAY_TASK_ID="${{SLURM_ARRAY_TASK_ID:-}}" \
        --env SLURM_PROCID="${{SLURM_PROCID:-0}}" \
        --env JOB_INDEX="$JOB_INDEX" \
        "$SIF_LOCAL" bash -lc '
            set -euo pipefail
            set -x
            echo "[container] SLURM_JOB_ID=${{SLURM_JOB_ID:-NA}} SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID:-NA}} SLURM_PROCID=${{SLURM_PROCID:-NA}} JOB_INDEX=${{JOB_INDEX:-NA}}"
            exec {PROJECT_DIR}/run.chunktask.sh
        '

        rc=$?
        echo "[entry] container exited with rc=$rc"
        exit $rc
    """))
    entry_sh.chmod(0o755)

    # launching command

    z1, dz2, dz3, dz4 = map(float, x)
    part     = os.environ.get("SLURM_PARTITION", "production")
    throttle = os.environ.get("ARRAY_MAX_CONCURRENCY", "32")

    array_cmd = (
        f"{SBATCH} --parsable "
        f"-p {part} "
        "--job-name=B0BO "
        f"--output={it_folder}/logs/chunk_%A_%a.out "
        f"--error={it_folder}/logs/chunk_%A_%a.err "
        "--ntasks=1 "
        f"--cpus-per-task={C} "
        "--time=02:00:00 "
        "--mem-per-cpu=4G " #"--mem=12G" !!! espace à la fin des lignes pour la suivante aha
        f"--array=0-{T-1}%{throttle} "
        f"--chdir={it_folder} "
        f"--export=NONE,PROJECT_DIR={PROJECT_DIR},NCHUNKS={T},SLURM_CPUS_PER_TASK={C},"
        f"EVENTS_PREFIX={os.environ.get('EVENTS_PREFIX','')},"
        f"DETECTOR_PATH={os.environ.get('DETECTOR_PATH','')},"
        f"DETECTOR_CONFIG={os.environ.get('DETECTOR_CONFIG','')},"
        f"ITER_DIR={it_folder},"
        f"Z1={z1},DZ2={dz2},DZ3={dz3},DZ4={dz4},"
        f"SIF_SRC={SIF_SHARED} "
        f"--wrap='bash {entry_sh.name}'"
    )

    # job

    jobid = _sbatch(array_cmd)
    print(f"[DEBUG simu.launch] Start: job sumbission {jobid}.")

    # waiting room

    final_state = _wait_array_done(jobid, poll_s=10)
    print(f"[DEBUG simu.launch] Achieved: job sumbission {jobid} with final code {final_state}.")

    # merging 

    simu_merged_file = simu_merger(T, C, x)
    if simu_merged_file is None:
        raise RuntimeError("[DEBUG simu.launch] ROOT merging too heavy, extra was ignored.")

    return simu_merged_file
