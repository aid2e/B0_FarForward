""" Utils functions for BOB0 """

# librairies

import os
import torch
torch.set_default_dtype(torch.double)
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ROOT
from pathlib import Path

# physics constants

MASS_LAMBDA = 1.1157 # GeV
MASS_PROTON = 0.9383 # GeV

# env global params

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR"))
EVENTS_PREFIX = os.environ.get("EVENTS_PREFIX")
OBS_OBJ = os.environ.get("OBS_OBJ")
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# env int reader

def env_int(name: str) -> int:
    """
    Reader of int environment variables.

    Returns:
        (int): env int value.
    """
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(f"[DEGUB utils.env_int] {name} absent.")
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"{name} must be integer.")

N_INIT = env_int("N_INIT")
N_ITER = env_int("N_ITER")

# env bounds reader

def get_bounds_from_env():
    """
    Reader of env upper and lower bounds.

    Returns:
        bounds (torch tensor): upper and lower bounds.
    """
    try:
        lower = json.loads(os.environ["LOWER_BOUNDS"])
        upper = json.loads(os.environ["UPPER_BOUNDS"])
    except KeyError as _:
        raise RuntimeError("c Missing bounds.")
    if len(lower) != len(upper):
        raise ValueError("[DEBUG utils.get_bounds_from_env] Different bounds size.")

    bounds = torch.tensor([lower, upper], dtype=torch.double)
    return bounds


# 4-vector construction function

def fourvec(px, py, pz, mass):
    """ 
    Build of a 4-vector from momenta and mass.

    Args:
        px (float): x-axis linear momentum [GeV].
        py (float): y-axis linear momentum [GeV].
        pz (float): z-axis linear momentum [GeV].
        mass (float): particle mass [GeV].

    Returns:
        (numpy array): 4-vector.
    """
    tot_energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.array([tot_energy, px, py, pz])


# 4-vector Minkowski norm

def fourvec_sqnorm(p):
    """
    Compute the Minkowski squared norm of a 4-vector.

    Args:
        p (numpy array): 4-vector [GeV].

    Returns:
        (float): Minkowski squared norm [GeV^2].
    """
    return p[0]**2 - (p[1]**2 + p[2]**2 + p[3]**2)


# Mandelstam variable from lambda 4-vector

def t_mandelstam(p1, p3):
    """
    Compute the second Mandelstam variable (t). 

    Args:
        p1 (numpy array): 4-vector proton [GeV].
        p3 (numpy array): 4-vector lambda [GeV].

    Returns:
        (float): second Mandelstam variable [GeV^2].
    """
    return fourvec_sqnorm(p1-p3)


# ROOT histogram filling

def th1(h, l):
    """
    Fill an histogram from a list of values.

    Args:
        l (array or list): list of values.
        h (TH1): ROOT histogram (1D).

    Returns:
        (TH1): ROOT histogram (1D).
    """

    # filling

    for _, val in enumerate(l):
        h.Fill(val)

    # style

    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(2)
    h.GetXaxis().CenterTitle(True)
    h.GetYaxis().CenterTitle(True)
    h.GetXaxis().SetTitleFont(132)
    h.GetYaxis().SetTitleFont(132)
    h.GetXaxis().SetLabelFont(132)
    h.GetYaxis().SetLabelFont(132)
    h.SetTitleFont(132, "XYZ")
    h.GetXaxis().SetTitleSize(0.05)
    h.GetYaxis().SetTitleSize(0.05)
    h.GetXaxis().SetLabelSize(0.04)
    h.GetYaxis().SetLabelSize(0.04)

    return h


# unitary cube de-normalization

def denormalize(x_norm, bounds):
    """
    Denormalization of geometrical parameters from the unitary cube to their bounds.
    
    Args:
        x_norm (float): Normalized geometrical parameter.
        bounds (Tensor 2D): Bounds of the geometrical parameter.
    
    Return:
        (float): Denormalized geometrical parameter.
    """
    lower = bounds[0]
    upper = bounds[1]
    return lower + (upper - lower) * x_norm

# unitary cube de-normalization

def normalize(x, bounds):
    """
    Normalization of gemetrical paramaters from their bounds to the unitary cube.
    
    Args:
        x (float): Real value of the geometrical parameter.
        bounds (Tensor 2D): Bounds of the geometrical parameter.
    
    Return:
        (float): Normalized geometrical parameter.
    """
    lower = bounds[0]
    upper = bounds[1]
    return (x - lower) / (upper - lower)

# plot convergence function

def plot_convergence(converg_fct, stat_err):
    """
    Plot the convergence function vs. iterations.
    
    Args:
        converg_fct (list float): Convergence metrics.
        stat_errr (list float): Statistial error 1/sqrt(n_hits).
    
    Return:
        filename (str): Plot file name.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(range(len(converg_fct)), 
                converg_fct, 
                yerr = stat_err,
                fmt='-s',  
                color='black',
                capsize=3, 
                ecolor='black')

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Convergence metric: max of SF")
    ax.set_title("Convergence function")

    filename = f'convergence_{EVENTS_PREFIX}.png'
    fig_dir = PROJECT_DIR / 'fig'
    plt.savefig(fig_dir/filename, bbox_inches='tight')
    plt.close(fig)
    return filename

# plot function for multi dimensional and single objective bayesian optimization

def plot_mobo(i, gp, bounds, x_train_norm, y_train):
    """
    Plot vizualization of Bayesian optimization at step i. 
    
    Args:
        i (int): Job number.
        gp (): Gaussian processes (Surrogate Function).
        bounds (Torch 2D): Bounds of geometrical parameters.
        x_train_norm (Torch): Bayesian trials (normalized).
        y_train (Torch): Bayesian evaluations.
    
    Return:
        filename (str): Saved plot path.
    """

    # dimensions

    dim = x_train_norm.shape[1]

    # denormalization

    x_train = denormalize(x_train_norm, bounds)

    # best point to this iteration

    best_idx = torch.argmax(y_train)
    best_x = x_train[best_idx]

    # figure

    fig, axes = plt.subplots(dim, dim, figsize=(4*dim, 4*dim))
    param_names = ['z1', 'dz2', 'dz3', 'dz4']
    n_points = 30

    # pair plot

    for line in range(dim):

        for row in range(dim):

            ax = axes[line, row]

            if line == row: # diagonals plots

                xi = torch.linspace(bounds[0, row], bounds[1, row], n_points)
                x_pd = best_x.repeat(n_points, 1).clone()
                x_pd[:, row] = xi
                x_pd_norm = (x_pd - bounds[0]) / (bounds[1] - bounds[0])
                posterior = gp.posterior(x_pd_norm)
                mean = posterior.mean.detach().cpu().numpy().squeeze()
                std = posterior.variance.sqrt().detach().cpu().numpy().squeeze()
                ax.plot(xi.cpu().numpy(), mean, color="black")
                ax.fill_between(xi.cpu().numpy(), mean-2*std, mean+2*std,color="black", alpha=0.2)
                ax.set_xticklabels([])
                ax.set_xlabel("")
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.tick_params(axis='x', direction='in')
                ax.tick_params(axis='y', direction='in')
                ax.set_ylabel("Partial Surrogate Function")

            elif line > row: # non-diagonals plot

                p_i = torch.linspace(bounds[0,row].item(), bounds[1,row].item(), n_points)
                p_j = torch.linspace(bounds[0,line].item(), bounds[1,line].item(), n_points)
                mesh_i, mesh_j = torch.meshgrid(p_i, p_j, indexing='ij')
                grid = best_x.repeat(n_points**2, 1).clone()
                grid[:, row] = mesh_i.flatten()
                grid[:, line] = mesh_j.flatten()
                grid_norm = (grid - bounds[0]) / (bounds[1] - bounds[0])
                mean = gp.posterior(grid_norm).mean.detach().cpu().numpy().reshape(n_points, n_points)
                ax.contourf(p_i, p_j, mean, levels=20, cmap='coolwarm')
                ax.scatter(x_train[:, row], x_train[:, line], c='k', s=10)

                # axis design

                if line == dim-1:
                    ax.set_xlabel(param_names[row])
                    ax.tick_params(axis='x', which='both', direction='in')
                    ax.tick_params(axis='y', which='both', direction='in')

                if line < dim-1:
                    ax.tick_params(axis='x',
                                   which='both',
                                   bottom=True,
                                   top=False,
                                   labelbottom=False,
                                   direction='in'
                                   )

                if row == 0:
                    ax.set_ylabel(param_names[line])
                    ax.tick_params(axis='x', which='both', direction='in')
                    ax.tick_params(axis='y', which='both', direction='in')

                if row > 0:
                    ax.tick_params(axis='y',
                                   which='both',
                                   left=True,
                                   right=False,
                                   labelleft=False,
                                   direction='in'
                                   )

                # best point in red

                ax.scatter(best_x[row].item(),best_x[line].item(),
                           color='red',
                           edgecolor='white',
                           s=80,
                           zorder=5
                           )

            else: # lower plot matrix
                ax.axis('off')

    # plot design

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
  
    # save & return

    fig_dir = PROJECT_DIR / 'fig'
    filename = f'B0_zopti_{EVENTS_PREFIX}_{i}.png'
    plt.savefig(fig_dir/filename)
    plt.close(fig)
    return filename
