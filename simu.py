""" Simulation launcher functions """

# libraries

import os, subprocess, json
import json, hashlib
from pathlib import Path
from geom import create_xml
import random
import shlex


# path managers


def _iter_tag(x):
    """
    Build a stable unique tag from geometry parameters.

    Args:
        x: Geometry parameters.

    Returns:
        str: Trial tag.
    """    
    if isinstance(x, dict):
        payload = [float(x[k]) for k in sorted(x.keys())]
    else:
        payload = [float(v) for v in x]
    s = json.dumps(payload, separators=(",", ":"))
    return "it_" + hashlib.sha1(s.encode()).hexdigest()[:10]


def _iter_dir(iter_tag):
    """
    Create and return the iteration directory.

    Args:
        iter_tag (str): Trial tag.

    Returns:
        Path: Iteration directory.
    """
    dir = Path(os.environ["AIDE_WORKDIR"]) / f"bayesian_iterations/{iter_tag}"
    dir.mkdir(parents=True, exist_ok=True)
    return dir


# new npsim


def run_npsim(b0_xml_job, iter_tag, n_events):
    """
    Run npsim in the container while overriding B0_tracker.xml via bind mount.

    Args:
        b0_xml_job (str | Path): Path to job-specific B0 tracker XML.
        iter_tag (str): Trial tag.
        n_events (int): Number of events to simulate.

    Returns:
        Path: Output ROOT file path.
    """
    it_dir = Path(_iter_dir(iter_tag)).resolve()
    it_dir.mkdir(parents=True, exist_ok=True)

    steering_file = (Path(os.environ["AIDE_HOME"]) / "steering.py").resolve()
    if not steering_file.exists():
        raise FileNotFoundError(f"steering.py not found: {steering_file}")

    b0_xml_job = Path(b0_xml_job).resolve()
    if not b0_xml_job.exists():
        raise FileNotFoundError(f"B0 job XML not found: {b0_xml_job}")

    output_file = it_dir / f"gun_proton_100GeV_{iter_tag}.root"
    log_file = it_dir / f"sim_log_{iter_tag}.out"
    err_file = it_dir / f"sim_log_{iter_tag}.err"

    seed = 1000 + random.randint(1, 10_000_000)

    singularity_exe = os.environ["SINGULARITY"]
    sif = os.environ["SIF"]

    detector_config = os.environ.get("DETECTOR_CONFIG", "epic_craterlake")

    # Probe DETECTOR_PATH inside container (after sourcing thisepic.sh)
    probe_cmd = (
        "set -euo pipefail; "
        "source /opt/detector/epic-main/bin/thisepic.sh; "
        'echo "$DETECTOR_PATH"'
    )
    probe_outer = (
        f"{shlex.quote(singularity_exe)} exec "
        f"{shlex.quote(sif)} /bin/bash -lc {shlex.quote(probe_cmd)}"
    )
    probe = subprocess.run(probe_outer, shell=True, capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"Failed to probe DETECTOR_PATH:\n{probe.stderr}\n{probe.stdout}")

    detector_path = probe.stdout.strip().rstrip("/")
    if not detector_path:
        raise RuntimeError("Empty DETECTOR_PATH from container probe")

    # Container-side paths used by EPIC geometry
    b0_in_container = f"{detector_path}/compact/far_forward/B0_tracker.xml"
    compact_in_container = f"{detector_path}/{detector_config}.xml"

    binds = [
        "/cvmfs:/cvmfs",
        "/srv:/srv",  # required for /srv/... paths in this environment
        f"{it_dir}:{it_dir}",
        f"{Path(os.environ['AIDE_HOME']).resolve()}:{Path(os.environ['AIDE_HOME']).resolve()}",
        f"{b0_xml_job}:{b0_in_container}",
    ]
    bind_flags = " ".join(f"--bind {shlex.quote(str(b))}" for b in binds)

    inner_cmd = (
        "set -euo pipefail; "
        "source /opt/detector/epic-main/bin/thisepic.sh; "
        'echo "[CONTAINER] DETECTOR_PATH=$DETECTOR_PATH"; '
        f'echo "[CONTAINER] Overriding: {b0_in_container}"; '
        f"if [ ! -f {shlex.quote(compact_in_container)} ]; then "
        '  echo "[CONTAINER][FATAL] Compact file not found. Listing candidates:" >&2; '
        f"  ls -1 {shlex.quote(detector_path)}/epic_*.xml 2>/dev/null || true; "
        "  exit 2; "
        "fi; "
        f"cd {shlex.quote(str(it_dir))}; "
        f"npsim --steeringFile {shlex.quote(str(steering_file))} "
        f"--random.seed {shlex.quote(str(seed))} "
        f"--numberOfEvents {shlex.quote(str(n_events))} --runType batch "
        f"--compactFile {shlex.quote(compact_in_container)} "
        f"--outputFile {shlex.quote(str(output_file))}"
    )

    outer_cmd = (
        f"{shlex.quote(singularity_exe)} exec "
        f"{bind_flags} "
        f"{shlex.quote(sif)} /bin/bash -lc {shlex.quote(inner_cmd)}"
    )

    print(f"[DEBUG simu.run_npsim] DETECTOR_PATH={detector_path}")
    print(f"[DEBUG simu.run_npsim] Bind B0: {b0_xml_job} -> {b0_in_container}")
    print(f"[DEBUG simu.run_npsim] Compact: {compact_in_container}")
    print(f"[DEBUG simu.run_npsim] Command: {outer_cmd}")

    with open(log_file, "w") as out, open(err_file, "w") as err:
        p = subprocess.run(outer_cmd, shell=True, stdout=out, stderr=err)

    print(
        f"[DEBUG simu.run_npsim] Error: simulation - {it_dir}. "
        f"Code {p.returncode}). See {log_file} and {err_file}."
    )

    # Print tail of logs (avoid dumping full files)
    try:
        err_txt = err_file.read_text(errors="replace").splitlines()
        out_txt = log_file.read_text(errors="replace").splitlines()
        print("======= BEGIN npsim STDERR (last 120 lines) =======")
        print("\n".join(err_txt[-120:]))
        print("======= END npsim STDERR =======")
        print("======= BEGIN npsim STDOUT (last 120 lines) =======")
        print("\n".join(out_txt[-120:]))
        print("======= END npsim STDOUT =======")
    except Exception as e:
        print(f"[ERROR] Could not read logs: {e}")

    if p.returncode != 0:
        raise RuntimeError(f"npsim failed with code {p.returncode}")

    print(f"[DEBUG simu.run_npsim] Achieved: simulation - {it_dir}.")
    return output_file


# simulation launching function


def launch(x: dict, n_events: int, cfg: dict):
    """
    Run one full simulation trial (geometry edit + npsim).

    Args:
        x: Bayesian trial parameters.
        n_events (int): Number of events to simulate.

    Returns:
        Path: Output ROOT file.
    """
    # iteration identifiers
    jobid = _iter_tag(x)
    print(f"[DEBUG simu.launch] Iter_tag / jobid : {jobid}.")

    # geometry modification
    print(f"[DEBUG simu.launch] Start: geometry modification - x={x}.")
    b0_xml_job = create_xml(x, jobid, cfg)
    print(f"[DEBUG simu.launch] Achieved: geometry modification - x={x}.")

    # simulation
    print(f"[DEBUG simu.launch] Start: simulation - x={x}.")
    outputfile = run_npsim(b0_xml_job, jobid, n_events)
    print(f"[DEBUG simu.launch] Achieved: simulation - x={x}.")

    return outputfile
