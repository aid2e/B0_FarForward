""" Run Bayesian optimization for B0 Far-Forward tracker design using Ax and PanDAiDDS """

import logging


# objective function


def objective_function(a, b, c, d, e, f, g, h, i):

    from simu import launch
    from analysis import meas_kin_in_container

    # simulation wrapper

    x_pos = [a, b, c, d]
    outputfile = launch(x_pos, 100)

    # objective

    x_mask = [e, f, g, h, i]
    obj = meas_kin_in_container(outputfile, x_mask)

    return {"objective": obj['p_err']}


# launch 


if __name__ == "__main__":
    
    from ax.service.ax_client import AxClient, ObjectiveProperties
    from scheduler import AxScheduler, PanDAiDDSRunner, JobLibRunner
    from scheduler.utils.common import setup_logging

    setup_logging(log_level="debug")

    logging.debug("setup ax client")

    # Initialize Ax client

    ax_client = AxClient()

    logging.info("Creating experiment")

    # Define your parameter space

    ax_client.create_experiment(
        name="my_experiment",
        parameters=[
            {
                "name": "a",
                "type": "range",
                "bounds": [20.0, 21.0], # cm
                "value_type": "float",
            },
            {
                "name": "b",
                "type": "range",
                "bounds": [20.0, 21.0], # cm
                "value_type": "float",
            },
            {
                "name": "c",
                "type": "range",
                "bounds": [20.0, 21.0], # cm
                "value_type": "float",
            },
            {
                "name": "d",
                "type": "range",
                "bounds": [20.0, 21.0], # cm
                "value_type": "float",
            },
            {
                "name": "e",
                "type": "range",
                "bounds": [-1.0, 1.0],
                "value_type": "float",
            },
            {
                "name": "f",
                "type": "range",
                "bounds": [-1.0, 1.0],
                "value_type": "float",
            },
            {
                "name": "g",
                "type": "range",
                "bounds": [-1.0, 1.0],
                "value_type": "float",
            },
            {
                "name": "h",
                "type": "range",
                "bounds": [-1.0, 1.0],
                "value_type": "float",
            },
            {
                "name": "i",
                "type": "range",
                "bounds": [-1.0, 1.0],
                "value_type": "float",
            },
        ],
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
        "max_walltime": 3600,
        "core_count": 1,
        "total_memory": 4000,
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

    best_params = scheduler.run_optimization(max_trials=2)
    print("Best parameters:", best_params)