# C. Fanelli 1-31-2022

import os,sys, pickle
import numpy as np
import torch # pytorch package, allows using GPUs
from platform import python_version
from psutil import virtual_memory, cpu_count
import torch
import numpy as np

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df  #https://ax.dev/api/service.html#ax.service.utils.report_utils.exp_to_df
from ax.runners.synthetic import SyntheticRunner

# Plotting imports and initialization
#from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
#init_notebook_plotting()

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

from ax import SumConstraint
from ax import OrderConstraint
from ax import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.parameter import RangeParameter,ParameterType

from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import ObjectiveThreshold, MultiObjectiveOptimizationConfig

from ax.core.experiment import Experiment

from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from ax.core.data import Data

from ax.core.types import ComparisonOp

from sklearn.utils import shuffle
from functools import wraps, lru_cache

from matplotlib import pyplot as plt

from matplotlib.cm import ScalarMappable

import timeit


#------------------------- RAM INFO ----------------------------#

def print_RAM_info():
  ram_gb = virtual_memory().total / 1e9
  print(f"Your runtime has {ram_gb:.1f} gigabytes of available RAM\n")

  ncpu = cpu_count()
  print(f"number of cpus: {ncpu}")

  if ram_gb < 20:
    print('Not using a high-RAM runtime')
  else:
    print('You are using a high-RAM runtime!')


def get_ram_usage():
    """
    Obtains the absolute number of RAM bytes currently in use by the system.
    :returns: System RAM usage in bytes.
    :rtype: int
    """

    #usage in MB
    memtot = int(virtual_memory().total/1024./1024.)
    memavail = int(virtual_memory().available/1024./1024.)
    res = memtot-memavail


    print(f"memory total (MB): {memtot}")
    print(f"memory available (MB): {memavail}")
    print(f"memory usage (MB): {res}")
    return res


#---------------------- TOY FUNCTIONS ------------------------#

def glob_dummy(loc_fun):
    @wraps(loc_fun)
    def inner(xdic):
        x_sorted = [xdic[p_name] for p_name in xdic.keys()] #it assumes x will be given as, e.g., dictionary
        x_sorted = np.array(x_sorted)
        res = loc_fun(x_sorted)
        return res

    return inner
# Define the objectives
@glob_dummy
def f1(x):
  res = np.sum(x*x)
  return res #obj1

@glob_dummy
def f2(x):
  res = 1./(np.sum(x*x)+0.01)
  return res #obj2
"""
def glob_fun(loc_fun):
  @lru_cache(maxsize=None)
  def inner(xdic):
    x_sorted = [xdic[p_name] for p_name in xdic.keys()]
    res = list(loc_fun(x_sorted))
    return res
  return inner
"""
def glob_fun(loc_fun):
    @wraps(loc_fun)
    @lru_cache(maxsize=None)
    def inner(xsorted):
      res = list(loc_fun(xsorted))
      return res
    return inner

#---------------------- BOTORCH FUNCTIONS ------------------------#

def build_experiment(search_space,optimization_config):
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment


def initialize_experiment(experiment,N_INIT):
    sobol = Models.SOBOL(search_space=experiment.search_space)

    experiment.new_batch_trial(sobol.gen(N_INIT)).run()

    return experiment.fetch_data()

def plot_scatter(df, outcomes, N_BATCH, BATCH_SIZE, N_INIT, n, outdir = "./"):
  fig, axes = plt.subplots(1, 1, figsize=(8,6))
  algos = ["qNEHVI"]
  train_obj = outcomes
  cm = plt.cm.get_cmap('viridis')
  n_results = N_BATCH*BATCH_SIZE + N_INIT
  batch_number = df.trial_index.values
  sc = axes.scatter(train_obj[:, 0], train_obj[:,1], c=batch_number, alpha=0.8)
  axes.set_title(algos[0])
  axes.set_xlabel("Objective 1")
  axes.set_ylabel("Objective 2")
  norm = plt.Normalize(batch_number.min(), batch_number.max())
  sm =  ScalarMappable(norm=norm, cmap=cm)
  sm.set_array([])
  fig.subplots_adjust(right=0.9)
  cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
  cbar = fig.colorbar(sm, cax=cbar_ax)
  cbar.ax.set_title("Iteration")
  plt.savefig(os.path.join(outdir, f"scatter_{n}.png"), dpi = 500)

def plot_hv(hv_list, n, outdir = "./"):
  fig, axes = plt.subplots(1, 1, figsize=(8,6))
  axes.plot(hv_list, "-o")
  axes.set_title("Hypervolume")
  axes.set_xlabel("Iteration")
  axes.set_ylabel("Hypervolume")
  plt.savefig(os.path.join(outdir, f"hv_{n}.png"), dpi = 500)
