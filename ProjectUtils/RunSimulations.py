from joblib import Parallel, delayed
import os, uuid, argparse

def RunSim(i: int, nEvents: int):
    steeringFile = os.environ.get("STEERING_FILE")
    parent_dir = os.environ.get("PARENT_DIR")
    run_dir = os.environ.get("RUN_DIR", os.path.join(parent_dir, str(uuid.uuid4())
                                                     )
                             )
    if (not steeringFile or not run_dir or not parent_dir):
        return -1
    else:
        temp = os.path.join(run_dir, f"ITER_{i}")
        os.makedirs(temp)
        os.chdir(temp)
        det_path = os.environ.get("DETECTOR_PATH")
        det_config = os.environ.get("DETECTOR")
        compactFile = os.path.join(det_path, det_config + ".xml")
        # npsim --steeringFile ../FromAlex/ddsim_steer_B0_testing.py --numberOfEvents 100 --compactFile ${DETECTOR_PATH}/${DETECTOR}.xml --outputFile FirstTry.edm4hep.root
        sim_cmd = f"npsim --steering {steeringFile} --numberOfEvents {nEvents} --compactFile {compactFile} --outputFile ResolutionStudy.edm4hep.root > sim_out.log 2>sim_err.log"
        print (sim_cmd)
        os.system(sim_cmd)
        # eicrecon -Pdd4hep:xml_files=${DETECTOR_PATH}/${DETECTOR}.xml -Ppodio_output_include_collections=ReconstructedParticles,GeneratedParticles,ReconstructedChargedParticles,BoTrackerRecHits -Pnthreads=8 FirstTry.edm4hep.root
        eicrecon_cmd = f"eicrecon -Pdd4hep:xml_files={compactFile} -Ppodio_output_include_collections=ReconstructedParticles,GeneratedParticles,ReconstructedChargedParticles,B0TrackerRecHits ResolutionStudy.edm4hep.root > recon_out.log 2>recon_err.log"
        print (eicrecon_cmd)
        os.system(eicrecon_cmd)
        ana_cmd = f"root -q -b \'{parent_dir}/SimplePlot.C(\"podio_output.root\")\'"
        print (ana_cmd)
        os.system(ana_cmd)
        return 0

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run simulations for the FarForward detector")
    argparser.add_argument("-n", "--numberOfEvents", type=int, default=50_000, help="Total Number of events to simulate")
    argparser.add_argument("-c", "--numberOfCores", type=int, default=20, help="Number of cores to use")
    argparser.add_argument("-s", "--steeringFile", type=str, help="Steering file to use", required=True)
    argparser.add_argument("-p", "--parentDir", type=str, help="Parent directory for the run", required=True)
    argparser.add_argument("-r", "--runDir", type=str, default="", help="Running directory")
    
    args = argparser.parse_args()
    nEvents = args.numberOfEvents
    nCores = args.numberOfCores
    nIterations = nEvents // nCores
    os.environ["STEERING_FILE"] = args.steeringFile
    os.environ["PARENT_DIR"] = args.parentDir
    if args.runDir != "":
        os.environ["RUN_DIR"] = args.runDir
        if (not os.path.exists(args.runDir)):
            os.makedirs(args.runDir)
    backend = "multiprocessing"
    outputs = Parallel(n_jobs=nCores, backend=backend, verbose = 10)(delayed(RunSim)(i, nIterations) for i in range(nCores))
    
    print (outputs)
    
