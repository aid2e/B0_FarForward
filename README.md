# **BOB0 : Bayesian Optimization for B0**  

## **Description**  

This repository provides a **Bayesian optimization module for B0**, a far-forward detector sub-system of ePIC [1], the future detector setup on EIC (Electron-Ion Collider) [2]. The optimized parameters are the **positions along the z-axis of the four tracker disks**. 

**The optimization is computationnaly intensive: the module is designed for running on iFarm with Slurm.** It also includes a checkpoint manager to start and stop at any time. The checkpoint files (to clean if you want to start from scratch) are saved in ***checkpoints/***.

## **Model & inputs**  

The Bayesian optimization [3], based on the BoTorch library [4], is based three functions:

- **Surrogate function (SF)**: fixed as Gaussian processes (GP). 
- **Acquisition function (AF)**: fixed as Noisy Expected Improvment (logNEI).
- **Objective function (OF)** : to be chosen by the user (see section Use).

The inputs, to be chosen by the user, are encoded in .hepevt files. They are stored in ***/events***, and include:

- **Proton gun at 41 GeV** (200k) along z-axis with 25 mrad tilting.
- **Proton gun at 100 GeV** (200k) along z-axis with 25 mrad tilting.
- **Proton gun at 275 GeV** (200k) along z-axis with 25 mrad tilting.
- **$\Lambda$ from Sullivan process at 5 x 41 GeV** (500k).
- **$\Lambda$ from Sullivan process at 10 x 100 GeV** (500k).
- **$\Lambda$ from Sullivan process at 18 x 275 GeV** (500k).

Remark: BOB0 simulations based on Geant4 manage instable particles and decay products, especially for $\Lambda$.

## **HPC strategy**  

**BOB0** divides the input file into **T x C** chunks, where **T** is the number of tasks and **C** the number of CPUs per task. Both **T** and **C** can be customize by the user (see section **Use**).

A volley of **T** small jobs are launched, each one managed by **C** CPUs, and saved in ***bayesian_iteration/***. Simulation results from all jobs are finally merged into a unique file (.root) saved in ***simu/***. This file will by analyzed to evaluate the objective function. 



## **Installation**

Installing **BOB0** is a 5-steps process. 

**Step 1.** Clone the repo on your personal working space in iFarm. It will create the project ***bob0/***.

```bash
git clone git@github.com:baptistefraisse/bob0.git
```

**Step 2.** Install **eic-shell** environment (the EIC group container) within the project.

```bash
cd bob0
mkdir eic
curl -L https://github.com/eic/eic-shell/raw/main/install.sh | bash
```

**Step 3.** Install an **old version of eic-shell** compatible with EICrecon on the EIC branch we'll work with. This step requires the **singularity** package (available on iFarm). Creation of the ***.sif*** file may be a long step (few hours). Once this file has been created, **change the very last line of .sif** with your project path. 

```bash
cd bob0
mkdir Old_eic-shell
mkdir $PWD/tmp && export SINGULARITY_CACHEDIR=$PWD
singularity pull --tmpdir=$PWD/tmp --name eic-jug_xl-nightly-2024-03-12.sif docker://eicweb/jug_xl@sha256:213e55fb304a92eb5925130cdff9529ea55c570b21ded9ec24471aa9c61219d8
cp ../eic/eic-shell ./custom-eic-shell
```

**Step 4.** Install the **ePIC model** for Geant4 simulations inside the custom-eic-shell environement. We'll work with a branch including a realistic magnetic field inside B0 called **b0-field-map-testing**. After cloning and before installing, **comments the lines 27 to 29 in CMakeLists.txt** (stop at any warning).

```bash
cd bob0
cd Old_eic-shell
source custom-eic-shell
git clone https://github.com/eic/epic.git -b b0-field-map-testing
cd epic # Here comment lines 27-28-29 in CMakeLists.txt
mkdir epic_install epic_build
cmake -B ./epic_build -S . -DCMAKE_INSTALL_PREFIX=./epic_install
cmake --build ./epic_build -j8
cmake --install ./epic_build
```

**Step 5.** Install **EICrecon**. We'll checkout the version **v1.11.0** compatible with our EIC environement (**custom-eic-shell**) and the **b0-field-map-testing** branch of ePIC.

```bash
cd bob0
cd Old_eic-shell
source custom-eic-shell
source ./epic_install/setup.sh
git clone https://github.com/eic/EICrecon.git
cd EICrecon
git checkout v1.11.0
mkdir EICrecon_build EICrecon_install
cmake -B ./EICrecon_build -S . -DCMAKE_INSTALL_PREFIX=./EICrecon_install
cmake --build ./EICrecon_build -j8
cmake --install ./EICrecon_build
```

## **Use**

Launching **BOB0** is a 3-steps process. The user interface lay in the ***bobo.env*** file.

**Step 1.** In ***bobo.env***, change the project direction afer the dash. For example, mine is: 

```bash
export PROJECT_DIR="${PROJECT_DIR:-/volatile/eic/fraisse/bob0}"
```

**Step 2.** In ***bobo.env***, customize the Bayesian optimization parameters: parallelization parameters, bounds, number of initialization and iterations, objective function and inputs. By default: 

```bash
export NARRAYS=100 # T
export NCPUS_PER_TASK=8 # C
export LOWER_BOUNDS='[0, 10, 10, 10]' # z1, dz2, dz3, dz4
export UPPER_BOUNDS='[40, 40, 40, 40]' # z1, dz2, dz3, dz4
export N_INIT=20 # number of init iterations
export N_ITER=50 # total number of iterations
export OBS_OBJ='p' # total momentum
export EVENTS_PREFIX=gun_proton_100GeV # 100 GeV protons
```

**Step 3.** Launch **BOB0** with Slurm using ***sbatch*** command with ***job.mobo.slurm.sh***. Status can be checked using ***squeue***. Jobs can be killed using ***scancel***.

```bash
sbatch job.mobo.slurm.sh
```

## **Outputs**

The progress of **BOB0** can be followed through two files, ***.out*** and ***.err***, written in dedicated folder ***logs/***. The final result (optimized z-positions) will be written at the very end of ***.out***. The figures, including the convergence function, are saved in the dedicated folder ***fig/***.




## **Outlooks**

Some improvements are planned or ongoing, including:

- [computational] Upgrade to multi-objective optimization (Paretto frontier).
- [computational] Upgrade to asynchronous Bayesian optimization (q > 1) [5].
- [physics] Optimization of active/dead zones of disks surfaces (Zernike decomposition).
- [physics] Holistic optimization of a physics channel (Sullivan process [6]).
- [cluster] Move from Slurm to Swif2.
- [cluster] Use of GPUs.


## **References**

[1] M. Pitt, Proc. of DIS2024, arXiv:2409.02811  (2024)

[2] Yellow Report on EIC, Nucl. Phys. A **1026**, 122447 (2022)

[3] B. Shahriari et al., Proc. of the IEEE (2016)

[4] M. Balandat et al., Advances in Neural Information Processing Systems (2020)

[5] B. Shahriari et al., Proc. of the IEEE (2016)

[6] J. D. Sullivan, Phys. Rev. D **5**, 1732 (1972)

## **Contact**

fraisse@cua.edu
