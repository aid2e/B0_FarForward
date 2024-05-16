import numpy as np
import os
import sys
import pandas as pd
from ProjectUtils.utils_optim import *
import math

from joblib import Parallel, delayed

import pickle
import dill
import uuid

import argparse

from timeit import default_timer as timer

import multiprocessing as mpr

import glob

#---------------------------------------------------------------------------------#


def whereAmI():
    #return os.path.dirname(os.path.realpath(__import__("__main__").__file__))
    return os.getcwd()

def RunSimInChunks(IterNo, designFile, SettingFile, events, etamin, etamax, pmin, pmax):
    FileSig = "TrackingOpt_Iter_{}.root".format(IterNo)
    print("Running the Iter : ", IterNo)
    CommandToExecute = "root -q -b"
    CommandToExecute += f" 'Fun4All_G4_EICDetector.C(\"{designFile}\", \"{SettingFile}\", {events}, {etamin}, {etamax}, {pmin}, {pmax}, \"{FileSig}\")'"
    timeout = "6m"
    success = os.system("timeout -k 10s " + timeout + " " + CommandToExecute + " > output_sim_{}.txt 2>error_sim_{}.txt".format(IterNo, IterNo))
    if(success == 0):
        return 0
    else:
        os.system("rm -rf TrackingOpt_Iter_{}_*.root".format(IterNo))
        print("timeout -k 10s " + timeout + " " + CommandToExecute + " > output_sim_{}.txt 2>error_sim_{}.txt".format(IterNo, IterNo))
        return -999

def RunSimInEta(designFile, SettingFile, threads, events_in_bins, eta_bins, momentum_range):
    counts = len(events_in_bins)
    backend = "multiprocessing"
    sim_out = Parallel(n_jobs=threads, backend = backend, verbose = 10)(delayed(RunSimInChunks)(th, designFile, SettingFile, events_in_bins[th], eta_bins[th][0], eta_bins[th][-1], momentum_range[0], momentum_range[-1]) for th in range(threads))
    len_files = len(glob.glob("TrackingOpt_Iter_*_g4tracking_eval.root"))
    if(len_files < 0.8*len(events_in_bins)): # disregard the simulation if total simulation has < 80%TotalEvents
        return -9999
    #success = os.system("hadd Combined_TrackingEval.root Iter_*/G4EICDetector_g4tracking_eval.root")
    success = os.system("hadd Combined_TrackingEval.root TrackingOpt_Iter_*_g4tracking_eval.root")
    success = 1
    return  "Combined_TrackingEval.root"

def RunSim(DesignParams, data, HowToRun = 0,  deldir=False, GenBaseline=False):

    #print("DesignParams : ", DesignParams)
    for k,v in DesignParams.items():
        print (f"## {k} : {v}")
    parent_dir = data["RUNNING_DIR"]

    parallel_dir = os.path.join(parent_dir, "parallel")

    if not os.path.exists(parallel_dir):
        os.makedirs(parallel_dir)

    #--------- working in /parallel
    os.chdir(parallel_dir)
    print(whereAmI())

    tmp_dir = str(uuid.uuid4())
    cmd = "mkdir "+str(tmp_dir)
    os.system(cmd)

    #--------- working in /parallel/tmp_dir
    working_dir = whereAmI()+'/'+str(tmp_dir)
    os.chdir(working_dir)
    
    # Create the design param file
    designFile = "design_params.config"
    with open(designFile, "w") as f:
        DesignParams1 = ReturnSummedParams(DesignParams, data["Signatures_NParams"])
        for k,v in DesignParams1.items():
            f.write(f"{k} : {v}\n")
            
    settingsFile = os.path.join(data["src_dir"], data["Settings"])
    cmd = "cp -r " + settingsFile + " ./CurrentSettings.setting"
    os.system(cmd)
    settingsFile = "CurrentSettings.setting"
    macros_dir = os.path.join(data["src_dir"], data["Macros"])
    cmd = "cp -r " + macros_dir + "/* ./"
    os.system(cmd)
    threads = data["N_Threads"]
    eta_bins = data["eta_bins"]
    N_tracks_per_event = data["N_tracks_per_event"]
    NoOfEvents = data["N_Events"]
    momentum_range = data["mom_range"]
    
    assert(threads >= len(eta_bins))
    EventsPerChunk = int((NoOfEvents/N_tracks_per_event)/threads) #5pions/event
    eta_bins = eta_bins*int(threads/len(eta_bins))
    events_in_bins = [EventsPerChunk]*len(eta_bins)
    
    print ("eta_bins : ", eta_bins)
    print ("events_in_bins : ", events_in_bins)
    print ("how to run", HowToRun)
    outputROOTFile= ""
    if (HowToRun == 1):
        NoOfEvents = -1
        outputROOTFile= SingularityExecute(data, designFile, settingsFile, NoOfEvents, [-3.4, 3.4], [1.0, 20.])
        sys.exit(1)
    if (HowToRun == 2):
        outputROOTFile= SingularityExecute(data, designFile, settingsFile, NoOfEvents, [-3.4, 3.4], [1.0, 20.])
    
    if (HowToRun == 0):
        outputROOTFile= RunSimInEta(designFile, settingsFile, threads, events_in_bins, eta_bins, momentum_range)


    commandAnalyse = "root -q -b 'analysis_resolution.C(\"{}\", {}, {})'".format("ExecuteSingularity_g4tracking_eval.root", int(GenBaseline), 1.4)
    print (commandAnalyse)
    #os.system(commandAnalyse + " > output_ana.txt 2>error_ana.txt")
    obj_r1  = CalcMeanMetric("dp_p_p", False)
    obj_r2  = CalcCostProxy(DesignParams1, "THICKNESS")
    obj_r3  = CalcMeanMetric("GlobalKFInEff", False) # This is KF_InEfficiency

    if((math.isnan(obj_r1)) or (math.isnan(obj_r2)) or (math.isnan(obj_r3))):
        obj_r1 = obj_r2 = obj_r3 = 1.

    
    os.chdir(parallel_dir)
    print(whereAmI())
    #print("##########")
    #exit()

    cmd = "rm -rf "+str(tmp_dir)
    if(deldir==True):

        os.system(cmd)
    os.chdir(parent_dir)

    print (obj_r1, obj_r2)
    return obj_r1, obj_r2


if __name__=="__main__":
    pass
