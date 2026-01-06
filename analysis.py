" Analysis of ROOT files from simulations """


# librairies


import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import ROOT
from utils import *
from pavement import B0_DISKS, zernike_mask_xy


# true kinematics


def true_kin():
    """
    True kinematics. 
    
    Returns: 
        results (dict): True kinematics variables.
    """
        
    filepath = PROJECT_DIR/f"events/edm4eic/{EVENTS_PREFIX}.edm4eic.root"
    tree = uproot.open(filepath)["events"]

    # observables of MCParticles (incident particles)

    pdg_array_in = tree["MCParticles.PDG"].array()
    px_array_in = tree["MCParticles.momentum.x"].array()
    py_array_in= tree["MCParticles.momentum.y"].array()
    pz_array_in = tree["MCParticles.momentum.z"].array()

    # observables of GeneratedParticles (outgoing particles)

    pdg_array_out = tree["GeneratedParticles.PDG"].array()
    px_array_out = tree["GeneratedParticles.momentum.x"].array()
    py_array_out = tree["GeneratedParticles.momentum.y"].array()
    pz_array_out = tree["GeneratedParticles.momentum.z"].array()

    # events scan

    proton_fourvec, lambda_fourvec, t_lambda = [], [], []
    p_proton_in, pt_proton_in = [], []
    p_lambda_out, pt_lambda_out = [], []

    for i in range(len(pdg_array_in)):

        # particles in

        pdg_in = pdg_array_in[i]
        px_in = px_array_in[i]
        py_in = py_array_in[i]
        pz_in = pz_array_in[i]
        proton_in_idxs = np.where(pdg_in == 2212)[0]
        proton_in_idx = proton_in_idxs[np.argmax(pz_in[proton_in_idxs])] # proton beam (max pz)

        proton_fourvec.append(fourvec(px_in[proton_in_idx], py_in[proton_in_idx], pz_in[proton_in_idx], MASS_PROTON))
        p_proton_in.append(np.sqrt(px_in[proton_in_idx]**2 + py_in[proton_in_idx]**2 + pz_in[proton_in_idx]**2))
        pt_proton_in.append(np.sqrt(px_in[proton_in_idx]**2 + py_in[proton_in_idx]**2))

        # particles out

        pdg_out = pdg_array_out[i]
        px_out = px_array_out[i]
        py_out = py_array_out[i]
        pz_out = pz_array_out[i]

        lambda_indexes = np.where(pdg_out == 3122)[0]
        lambda_index = lambda_indexes[0] # the only one

        lambda_fourvec.append(fourvec(px_out[lambda_index], py_out[lambda_index], pz_out[lambda_index], MASS_LAMBDA))
        p_lambda_out.append(np.sqrt(px_out[lambda_index]**2 + py_out[lambda_index]**2 + pz_out[lambda_index]**2))
        pt_lambda_out.append(np.sqrt(px_out[lambda_index]**2 + py_out[lambda_index]**2))

        # mandelstam

        t_lambda.append(t_mandelstam(fourvec(px_in[proton_in_idx], py_in[proton_in_idx], pz_in[proton_in_idx], MASS_PROTON), 
                                     fourvec(px_out[lambda_index], py_out[lambda_index], pz_out[lambda_index], MASS_LAMBDA)))
        
    # histograms

    hist_p_proton_in = ROOT.TH1F("hist_p_proton_in", "hist_p_proton_in; p [GeV]; Counts", 2000, -2000, 2000)
    hist_p_proton_in = th1(hist_p_proton_in, p_proton_in)

    hist_pt_proton_in = ROOT.TH1F("hist_pt_proton_in", "hist_pt_proton_in; p [GeV]; Counts", 100, 0, 200)
    hist_pt_proton_in = th1(hist_pt_proton_in, pt_proton_in)

    hist_p_lambda_out = ROOT.TH1F("hist_p_lambda_out", "hist_p_lambda_out; p [GeV]; Counts", 100, 0, 200)
    hist_p_lambda_out = th1(hist_p_lambda_out, p_lambda_out)

    hist_pt_lambda_out = ROOT.TH1F("hist_pt_lambda_out", "hist_pt_lambda_out; pT [GeV]; Counts", 100, 0, 200)
    hist_pt_lambda_out = th1(hist_pt_lambda_out, pt_lambda_out)

    hist_t_lambda = ROOT.TH1F("hist_t_lambda", "hist_t_lambda; t [GeV^2]; Counts", 100, -0.06, 0.01)
    hist_t_lambda = th1(hist_t_lambda, t_lambda)

    # results dict

    results = {
        "p_proton_in":   hist_p_proton_in.GetMean(),
        "pt_proton_in":  hist_pt_proton_in.GetMean(),
        "p_lambda_out":  hist_p_lambda_out.GetMean(),
        "pt_lambda_out": hist_pt_lambda_out.GetMean(),
        "t_lambda":      hist_t_lambda.GetMean(),
    }

    return results


# measured kinematic


def meas_kin(merged_file):
    """
    Measured kinematics.

    Args:
        merged_file (str): Geant4 simulation results after merging (ROOT).
    
    Returns: 
        results (dict): Measured kinematics in B0 disks (p, pT).
    """

    with uproot.open(merged_file) as file:

        # read

        tree = file["events"]

        x = tree["B0TrackerHits.position.x"].arrays(library="ak")["B0TrackerHits.position.x"]
        y = tree["B0TrackerHits.position.y"].arrays(library="ak")["B0TrackerHits.position.y"]
        z = tree["B0TrackerHits.position.z"].arrays(library="ak")["B0TrackerHits.position.z"]

        px = tree["B0TrackerHits.momentum.x"].arrays(library="ak")["B0TrackerHits.momentum.x"]
        py = tree["B0TrackerHits.momentum.y"].arrays(library="ak")["B0TrackerHits.momentum.y"]
        pz = tree["B0TrackerHits.momentum.z"].arrays(library="ak")["B0TrackerHits.momentum.z"]

        # kinematics

        p = np.sqrt(px**2 + py**2 + pz**2)
        pt = np.sqrt(px**2 + py**2)

        # awkward distribution reconstruction

        p_flat = ak.flatten(p)
        pt_flat = ak.flatten(pt)

    # hist

    hist_p_awkward = ROOT.TH1F("hist_p_awkward", "p distribution; p [GeV]; Counts", 100, 0, 1)
    hist_p_awkward = th1(hist_p_awkward, p_flat)
    hist_pt_awkward = ROOT.TH1F("hist_pt_awkward", "pt distribution; pt [GeV]; Counts", 100, 0, 1)
    hist_pt_awkward = th1(hist_pt_awkward, pt_flat)

    # dict results

    results = {
        "p":  hist_p_awkward.GetMean(),
        "p_err": hist_p_awkward.GetMeanError(),
        "pT": hist_pt_awkward.GetMean(),
        "pT_err": hist_pt_awkward.GetMeanError()
    }

    return results


# def meas_kin_masked(merged_file, c, N=2, sigma=0.8):
#     """
#     Measured kinematics by B0 disks with mask.

#     Args:
#         merged_file (str): Geant4 simulation results after merging (ROOT).
#         c (list float): Zernike polynomials linear combination coefficients.
#         N (int): Zernike polynomials order cut-off.
#         sigma (float): Total active ratio of B0 disks.
    
#     Returns: 
#         results (dict): Measured kinematics in B0 disks (p, pT).
#     """

#     with uproot.open(merged_file) as file:

#         tree = file["events"]

#         x = tree["B0TrackerHits.position.x"].arrays(library="ak")["B0TrackerHits.position.x"]
#         y = tree["B0TrackerHits.position.y"].arrays(library="ak")["B0TrackerHits.position.y"]
#         z = tree["B0TrackerHits.position.z"].arrays(library="ak")["B0TrackerHits.position.z"]

#         px = tree["B0TrackerHits.momentum.x"].arrays(library="ak")["B0TrackerHits.momentum.x"]
#         py = tree["B0TrackerHits.momentum.y"].arrays(library="ak")["B0TrackerHits.momentum.y"]
#         pz = tree["B0TrackerHits.momentum.z"].arrays(library="ak")["B0TrackerHits.momentum.z"]

#         # disk geom

#         rmin = B0_DISKS[0]["Rint"]
#         rmax = B0_DISKS[0]["Rext"]
#         cx = B0_DISKS[0]["x0"] 
#         cy = B0_DISKS[0]["y0"] 
#         xmin, xmax = cx - rmax, cx + rmax
#         ymin, ymax = cy - rmax, cy + rmax
#         x_disk = np.linspace(xmin, xmax, res)
#         y_disk = np.linspace(ymin, ymax, res)
#         X, Y = np.meshgrid(x, y)

#         # mask

#         mask = zernike_mask_xy(c=c, N=N, sigma=sigma, x=X, y=Y, rmax=rmax, rmin=rmin, center=(cx, cy))
#         keep = (mask == 1)
#         px = ak.where(keep, px, 0.0)
#         py = ak.where(keep, py, 0.0)
#         pz = ak.where(keep, pz, 0.0)

#         # kinematics of interest

#         p = np.sqrt(px**2 + py**2 + pz**2)
#         pt = np.sqrt(px**2 + py**2)
#         p_flat = ak.flatten(p)
#         pt_flat = ak.flatten(pt)

#     # hist

#     hist_p_awkward = ROOT.TH1F("hist_p_awkward", "p distribution; p [GeV]; Counts", 100, 0, 1)
#     hist_p_awkward = th1(hist_p_awkward, p_flat)
#     hist_pt_awkward = ROOT.TH1F("hist_pt_awkward", "pt distribution; pt [GeV]; Counts", 100, 0, 1)
#     hist_pt_awkward = th1(hist_pt_awkward, pt_flat)

#     results = {
#         "p":  hist_p_awkward.GetMean(),
#         "pT": hist_pt_awkward.GetMean(),
#     }

#     return results


# reconstructed kinematic (electron method from EICrecon)


def recon_kin(merged_file):
    """
    Reconstructed kinematics.

    Args:
        merged_file (str): Geant4 simulation results after merging (ROOT).
    
    Returns: 
        results (dict): Reconstructed kinematics (p, pT).
    """

    with uproot.open(merged_file) as file:

        # read

        tree = file["events"]
        px = tree["ReconstructedParticles.momentum.x"].arrays(library="ak")["ReconstructedParticles.momentum.x"]
        py = tree["ReconstructedParticles.momentum.y"].arrays(library="ak")["ReconstructedParticles.momentum.y"]
        pz = tree["ReconstructedParticles.momentum.z"].arrays(library="ak")["ReconstructedParticles.momentum.z"]

        # kinematics

        p = np.sqrt(px**2 + py**2 + pz**2)
        pt = np.sqrt(px**2 + py**2)

        # awkward distribution reconstruction

        p_flat = ak.flatten(p)
        pt_flat = ak.flatten(pt)

    # hist

    hist_p_awkward = ROOT.TH1F("hist_p_awkward", "p distribution; p [GeV]; Counts", 100, 0, 1)
    hist_p_awkward = th1(hist_p_awkward, p_flat)
    hist_pt_awkward = ROOT.TH1F("hist_pt_awkward", "pt distribution; pt [GeV]; Counts", 100, 0, 1)
    hist_pt_awkward = th1(hist_pt_awkward, pt_flat)

    # dict results

    results = {
        "p":  hist_p_awkward.GetMean(),
        "pT": hist_pt_awkward.GetMean(),
    }

    return results
