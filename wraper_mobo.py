import ax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import ObjectiveThreshold, MultiObjectiveOptimizationConfig
from ax.core.types import ComparisonOp
from ax.core.search_space import SearchSpace
from ax.core.parameter import RangeParameter,ParameterType
from ax.modelbridge.registry import Models
from ax.core.data import Data
from ax.service.utils.report_utils import exp_to_df
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from psutil import virtual_memory, cpu_count
import torch
from ax.core.experiment import Experiment
from ax.runners.synthetic import SyntheticRunner
import pickle
import argparse
from ProjectUtils.mobo_utilities import *
from ProjectUtils.mobo_objectives import *
from ProjectUtils.config_editor import *
from ProjectUtils.runSimulator import *
  

if __name__ == "__main__":
  pd.set_option('display.max_columns', None)
  parser = argparse.ArgumentParser(description= "Wrapper MOBO for Tracker optimization")
  parser.add_argument('-c', '--MOBOConfig', help='MOBO configuration file', type = str, required = True)
  parser.add_argument('-loadpkl','--loadpkl', help=' load existing pickle', type = bool, default=False, required = False)
  parser.add_argument('-dummy','--dummy', help=' use dummy simulation', type = bool, default=False, required = False)
  parser.add_argument('-json_file', '--json_file', help = "The json file to load", type = str, required=False)
  args = parser.parse_args()
  
  # READ SOME INFO 
  config = ReadJsonFile(args.MOBOConfig)
  N_THREADS = config["N_Threads"]
  N_BATCH = config["N_Calls"]
  BATCH_SIZE = config["Batch_Size"]
  want_load_pkl = False
  dim_space = config["N_DesignParams"]
  UseDummy = False
  save_every_n = config["save_period"]
  outdir = config["save_dir"]
  save_prefix = config["save_prefix"] #hard-coded
  settings = config["Settings"]
  os.environ["SETTINGS"] = settings
  
  if (args.loadpkl == True and args.json_file == None):
    print ("ERROR: you need to specify the json file to load when using the option loadpkl")
    sys.exit(1)
  if (args.loadpkl == True and os.path.isfile(args.json_file) == False):
    print ("ERROR: the json file you specified does not exist")
    sys.exit(1)
  
  print ("####### OPTIMIZATION INFORMATION ###########")
  print (f"## DIM SPACE : {dim_space}")
  print (f"## N_BATCH / N_CALLS : {N_BATCH}")
  print (f"## BATCH_SIZE : {BATCH_SIZE}")
  print (f"## LOAD PKL : {want_load_pkl}")
  print (f"## USE DUMMY : {UseDummy}")
  print (f"## SAVE EVERY N : {save_every_n}")
  print (f"## N_THREADS : {N_THREADS}")
  print (f"## OUTDIR : {outdir}")
  print ("############################################")
  
  
  
  # initial number of samples to explore
  N_INIT = 2 * (dim_space + 1) 
  print("N_INIT, N_BATCH, BATCH_SIZE: ", N_INIT,N_BATCH,BATCH_SIZE)
  exp_evaluations = N_INIT + N_BATCH*BATCH_SIZE
  print ("Expected N. Objectives Evaluations : ", exp_evaluations)

  tkwargs = {
          "dtype": torch.double,
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
      } 
  
  params = GetDesignParamNames(config["Signatures_NParams"], config["DesignParams_Ranges"])
  for k,v in params.items():
    print (f"## {k} : {v}")
  assert(len(params) == dim_space)
  lowerv =[ 0.]*dim_space 
  upperv =[ 1.1]*dim_space
  ref_f1 = 1. # This has to be changed.
  ref_f2 = 1.
  search_space = None
    
  
  if(UseDummy==True):
    print("\n\n\n############### DUMMY TEST ###############")
    @glob_dummy
    def f1(x):
      global res
      res = np.sum(x*x)
      return res #obj1
    @glob_dummy
    def f2(x):
      global res
      res = 1./(np.sum(x*x)+0.01)
      return res #obj2
    search_space = SearchSpace(RangeParameter(name=f"x{i}", lower=lowerv[i], upper=upperv[i], 
                                              parameter_type=ParameterType.FLOAT) 
                               for i in range(dim_space)
                               )
                               
        
    
  elif(UseDummy==False):
    print("\n\n\n############### F4A SIMULATION ###############")
    @glob_fun
    def ftot(x):
        return RunSim(x, config) #obj1, obj2, obj3, obj4, g1, g2, g3
    def f1(xdic):
      return ftot(xdic)[0] #obj1
    def f2(xdic):
      return ftot(xdic)[1] #obj2  
  
    search_space = SearchSpace(
        parameters=[RangeParameter(name=key, 
                                    lower=value["min"], upper=value["max"], 
                                    parameter_type=ParameterType.FLOAT
                                    ) 
                    for key, value in params.items()
                    ],
        #parameter_constraints=[con_1, con_2, con_3] # add your constraints...
    )


  #lower_is_better=True if you are minimizing
  metric_a = GenericNoisyFunctionMetric("mom_resolution", f=f1, noise_sd=0.0, lower_is_better=True)
  metric_b = GenericNoisyFunctionMetric("costProxy", f=f2, noise_sd=0.0, lower_is_better=True)

  mo = MultiObjective(
      objectives=[Objective(metric=metric_a), Objective(metric=metric_b)],
  )

  refpoints = torch.Tensor([ref_f1, ref_f2]).to(**tkwargs)   #we want to minimize both functions

  print("refpoints: ", refpoints)

  # see https://github.com/facebook/Ax/blob/main/ax/core/tests/test_optimization_config.py
  # relative=False -> obj taken as absolute; True -> relative to obj
  # op=ComparisonOp.LEQ (request obj <= threshold)
  # Make sure the thresholds are above 0. otherwise, it makes no sense
  objective_thresholds = [
      ObjectiveThreshold(metric=metric, bound=val*0., relative=False, op=ComparisonOp.LEQ)
      for metric, val in zip(mo.metrics, refpoints) #---> this requires defining a torch.float64 object --- by default is (-)1.1 for DTLZ
      ]

  optimization_config = MultiObjectiveOptimizationConfig(
      objective=mo,
      objective_thresholds=objective_thresholds,
  )

  model = None
  experiment = None
  data = None
  outcomes = None

  hv_list = []
  last_call = 0
  last_mem = -1.
  if(want_load_pkl == False):
    experiment = build_experiment(search_space,optimization_config)
    data = initialize_experiment(experiment,N_INIT)
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[['a', 'b']].values, **tkwargs)
    partitioning = DominatedPartitioning(ref_point=refpoints, Y=outcomes)
    try:
      hv = partitioning.compute_hypervolume().item()
    except:
      hv = 0
      print("Failed to compute hv")
    hv_list.append(hv)
    print(f"Initialized points, HV: {hv}")
    with open(os.path.join(outdir, "ax_state_init.json"), 'wb') as handle:
      list_dump = [-1, experiment, np.nan, last_mem, data, outcomes]
      pickle.dump(list_dump, handle, pickle.HIGHEST_PROTOCOL)
      print("saved a file")
  if (want_load_pkl == True): 
    print("\n\n WARNING::YOU ARE LOADING AN EXISTING FILE: ", bo_save_path, "\n\n")
    tmp_list = pickle.load(open(bo_save_path, "rb" ))
    last_call = tmp_list[0] + 1
    experiment = tmp_list[1]
    hv_list = tmp_list[2]
    last_mem =  tmp_list[3]
    data = tmp_list[4]
    outcomes = tmp_list[5]

  batch_range = np.arange(N_BATCH)+last_call
  it_b = iter(batch_range)

  #--------------------------------------------------------------------------#
  #https://ax.dev/tutorials/saasbo_nehvi.html
  #while (i := next(it_b, None)) is not None:
  while (i < N_BATCH):
    print("\n\n...PROCESSING BATCH n.: {}\n\n".format(i+1))
    # track
    print('track memory usage before model')
    tmpmem = get_ram_usage()

    model = Models.FULLYBAYESIANMOO(
        experiment=experiment,
        data=data, # tell the data
        # use fewer num_samples and warmup_steps to speed up this tutorial
        num_samples=128,#256
        warmup_steps=256,#512
        torch_device=tkwargs["device"],
        verbose=False,  # Set to True to print stats from MCMC
        disable_progbar=False,  # Set to False to print a progress bar from MCMC
        )

    generator_run = model.gen(BATCH_SIZE)   #ask BATCH_SIZE points
    trial = experiment.new_batch_trial(generator_run=generator_run)
    trial.run()
    data = Data.from_multiple_data([data, trial.fetch_data()])   #https://ax.dev/api/core.html#ax.Data.from_multiple_data
    ftot.cache_clear() # Just in case 
    
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[['a', 'b']].values, **tkwargs)
    partitioning = DominatedPartitioning(ref_point=refpoints, Y=outcomes)
    try:
      hv = partitioning.compute_hypervolume().item()
    except:
      hv = 0
      print("Failed to compute hv")
    hv_list.append(hv)
    print(f"Iteration: {i+1}, HV: {hv}")
    if i % save_every_n == 0:
        with open(os.path.join(outdir, save_prefix + f'_iteration_{i}.json'), 'wb') as handle:
            list_dump = [i, experiment, hv_list, tmpmem, data, outcomes]
            pickle.dump(list_dump, handle)
            print("saved a file")
            plot_scatter(exp_df, outcomes, N_BATCH, BATCH_SIZE, N_INIT, i, outdir)
            plot_hv(hv_list, i, outdir)
