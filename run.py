""" Run Bayesian optimization for B0 Far-Forward tracker design using Ax and PanDAiDDS """

import logging
import yaml

# loading yaml

with open("b0_params.yml") as f:
    CFG = yaml.safe_load(f)

# objective function

def objective_function(**parameterization):

    import os
    from simu import launch
    from analysis import meas_kin_in_container
    from geom import create_xml

    geom_values = {
        "z1" : parameterization["b0_tracker.z1"] ,
        "dz2": parameterization["b0_tracker.dz2"],
        "dz3": parameterization["b0_tracker.dz3"],
        "dz4": parameterization["b0_tracker.dz4"],
    }

    jobid = os.environ.get("PANDA_JOB_ID", "local")
    b0_xml_job = create_xml(geom_values, jobid, CFG)

    outputfile = launch(geom_values, int(CFG["simulation"]["n_events"]), CFG)

    x_mask = [
        parameterization.get("analysis_mask.e", 0.0),
        parameterization.get("analysis_mask.f", 0.0),
        parameterization.get("analysis_mask.g", 0.0),
        parameterization.get("analysis_mask.h", 0.0),
        parameterization.get("analysis_mask.i", 0.0),
    ]

    obj = meas_kin_in_container(outputfile, x_mask)
    
    return {"objective": obj["p_err"]}

# launch 

if __name__ == "__main__":
    
    from ax.service.ax_client import AxClient, ObjectiveProperties
    from scheduler import AxScheduler, PanDAiDDSRunner, JobLibRunner
    from scheduler.utils.common import setup_logging
    from param_config import build_ax_parameters

    setup_logging(log_level="debug")

    logging.debug("setup ax client")

    group_name = "b0_geom_and_mask"
    ax_client = AxClient()
    ax_client.create_experiment(
        name="my_experiment",
        parameters=build_ax_parameters(CFG, group_name),
        objectives={"objective": ObjectiveProperties(minimize=True)},
    )

    logging.info("defining objectives")
    
    init_env = [
        'source /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/wguan/mlcontainer:py311_1.0/opt/conda/setup_mamba.sh;'
        'source /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/wguan/mlcontainer:py311_1.0/opt/conda/dRICH-MOBO/MOBO-tools/setup_new.sh;'
        'command -v singularity &> /dev/null || '
        'export SINGULARITY=/cvmfs/oasis.opensciencegrid.org/mis/singularity/current/bin/singularity;'
        'unset SINGULARITY_BINDPATH;'
        'export AIDE_HOME=$(pwd);'
        'export PWD_PATH=$(pwd);'
        'export EPIC_SHARE_LOCAL="$PWD/epic_share_no_colon";'
        'mkdir -p "$EPIC_SHARE_LOCAL";'
        'if [ ! -e "$EPIC_SHARE_LOCAL/epic" ]; then ln -s "${DETECTOR_PATH}" "$EPIC_SHARE_LOCAL/epic"; fi;'
        'export SIF=/cvmfs/singularity.opensciencegrid.org/eic/eic_ci:nightly; '
        'env; '
    ]

    init_env = " ".join(init_env)

    panda_attrs = {
        "name": "user.wguan.my_experiment",
        "init_env": init_env,
        "cloud": "US",
        "queue": "BNL_PanDA_1",  # BNL_OSG_PanDA_1, BNL_PanDA_1
        "source_dir": None,
        "source_dir_parent_level": 1,
        "exclude_source_files": [
            r"(^|/)\.[^/]+",    # file starts with "."
            "doc*", "DTLZ2*", ".*json", ".*log", "work", "log", "OUTDIR",
            "calibrations", "fieldmaps", "gdml", "EICrecon-drich-mobo",
            "eic-software", "epic-geom-drich-mobo", "irt", "share", "back*",
            "__pycache__", "events", "checkpoints"
        ],
        "max_walltime": int(CFG["simulation"]["max_walltime"]),
        "core_count": int(CFG["simulation"]["core_count"]),
        "total_memory": int(CFG["simulation"]["total_memory"]),
        "enable_separate_log": True,
        "job_dir": None,
    }

    # Create a runner

    runner =  PanDAiDDSRunner(**panda_attrs) 
    logging.info(f"created runner: {runner}")

    # Create the scheduler

    scheduler = AxScheduler(ax_client, runner)
    logging.info(f"created scheduler: {scheduler}")

    # Set the objective function

    scheduler.set_objective_function(objective_function)
    logging.info("running optimization")
    
    # Run the optimization

    best_params = scheduler.run_optimization(max_trials=int(CFG["simulation"]["n_trials"]))
    print("Best parameters:", best_params)