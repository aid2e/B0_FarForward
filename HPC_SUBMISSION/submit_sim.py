#!/usr/bin/env python

import os, argparse, json, uuid

def main(args):
    # Get the PBSCONFIG info
    with open(args.configFile, "r") as f:
        pbsconfig = json.load(f)
    simulation = pbsconfig["simulation"]
    sim_name = simulation["name"]
    n_events = int(simulation["nEvents"])
    n_jobs = int(simulation["nJobs"])
    script_dir = pbsconfig["input"]["CodeDir"]
    run_sim_script = os.path.join(script_dir, "run_sim.sh")
    submit_script = os.path.join(script_dir, "SUBMIT.csh")
    container = pbsconfig["container"]["path"]
    
    outDir = pbsconfig["output"]["outDir"]
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    workDir = pbsconfig["output"].get("workDir", outDir)
    if not os.path.exists(workDir):
        os.makedirs(workDir)
    logDir = pbsconfig["output"]["logDir"]
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    epic_install = pbsconfig["epic"]["epicDir"]
    eicrecon_install = pbsconfig["eicrecon"]["eicreconDir"]
    
    for it in range(n_jobs):
        uuid_str = str(uuid.uuid4())
        jobName = f"{sim_name}_{it}"
        job_dir = os.path.join(workDir, f"ITER_{it}")
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        # Modify SUBMIT.sh script
        with open(submit_script, "r") as f:
            submit_contents = f.read()
        submit_contents = submit_contents.replace("JOB_NAME", jobName)
        submit_contents = submit_contents.replace("WORK_DIR", job_dir)
        submit_contents = submit_contents.replace("LOG_DIR", logDir)
        submit_contents = submit_contents.replace("OUTPUT_DIR", job_dir)
        submit_contents = submit_contents.replace("EIC_SHELL", container)
        submit_contents = submit_contents.replace("SCRIPTFILE", os.path.join(job_dir, "run_sim.sh"))
        
        with open(f"{job_dir}/SUBMIT.csh", "w") as f:
            f.write(submit_contents)
            
        # Modify run_sim.sh script
        with open(run_sim_script, "r") as f:
            run_sim_contents = f.read()
        run_sim_contents = run_sim_contents.replace("epic_install", epic_install)
        run_sim_contents = run_sim_contents.replace("eicrecon_install", eicrecon_install)
        run_sim_contents = run_sim_contents.replace("code_dir", script_dir)
        run_sim_contents = run_sim_contents.replace("n_events", str(n_events))
        run_sim_contents = run_sim_contents.replace("out_dir", job_dir)
        
        with open(f"{job_dir}/run_sim.sh", "w") as f:
            f.write(run_sim_contents)
        
        os.system("qsub " + os.path.join(job_dir, "SUBMIT.csh"))
        
        
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Submit simulations for the FarForward detector")
    argparser.add_argument("-c", "--configFile", 
                           help = "Path to the configuration file for the simulation", 
                           type = str, required = True
                           )
    args = argparser.parse_args()
    main(args)