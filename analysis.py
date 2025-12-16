" Analysis of ROOT files from simulations """

# histogram utils

def th1(h, l):
    """
    Fill an histogram from a list of values.

    Args:
        l (array or list): list of values.
        h (TH1): ROOT histogram (1D).

    Returns:
        (TH1): ROOT histogram (1D).
    """

    import ROOT

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

# measured kinematic


def meas_kin(outputfile, mask=None):
    """
    Measured kinematics.

    Args:
        merged_file (str): Geant4 simulation results after merging (ROOT).
        mask (callable, optional): A function that takes x and y coordinates and returns a boolean mask.

    Returns:
        results (dict): Measured kinematics in B0 disks (p, pT).
    """

    import uproot
    import numpy as np
    import awkward as ak
    import ROOT

    with uproot.open(outputfile) as file:

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

        x_flat  = ak.to_numpy(ak.flatten(x))
        y_flat  = ak.to_numpy(ak.flatten(y))
        z_flat = ak.flatten(z)
        p_flat = ak.flatten(p)
        pt_flat = ak.flatten(pt)

        # deadzones mask

        if mask is not None:
            keep_mask = np.asarray(mask(x_flat, y_flat), dtype=bool)
            finite_mask = np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(p_flat) & np.isfinite(pt_flat)
            keep = np.logical_and(keep_mask, finite_mask)
            x_flat  = x_flat[keep]
            y_flat  = y_flat[keep]
            p_flat  = p_flat[keep]
            pt_flat = pt_flat[keep]

    # hist

    hist_p_awkward = ROOT.TH1F("hist_p_awkward", "p distribution; p [GeV]; Counts", 100, 0, 1)
    hist_p_awkward = th1(hist_p_awkward, p_flat)
    hist_pt_awkward = ROOT.TH1F("hist_pt_awkward", "pt distribution; pt [GeV]; Counts", 100, 0, 1)
    hist_pt_awkward = th1(hist_pt_awkward, pt_flat)
    hist_z_awkward = ROOT.TH1F("hist_z_awkward", "z distribution; z [cm]; Counts", 7000, 0, 7000)
    hist_z_awkward = th1(hist_z_awkward, z_flat)

    # dict results

    results = {
        "p":  hist_p_awkward.GetMean(),
        "p_err": hist_p_awkward.GetMeanError(),
        "pT": hist_pt_awkward.GetMean(),
        "pT_err": hist_pt_awkward.GetMeanError(),
        "z_av": hist_z_awkward.GetMean()
    }

    return results

def meas_kin_in_container(root_file, x_mask=None):
    """
    Run analysis_cli.py on a ROOT file inside the container and return JSON dict.

    Args:
        root_file (str | Path): Input ROOT file.
        x_mask (list[float] | None): Mask coefficients forwarded to analysis_cli.py.

    Returns:
        dict: Parsed JSON output from analysis_cli.py.
    """
    import os, subprocess, shlex, json

    root_file = str(root_file)
    workdir = os.environ.get("AIDE_HOME", os.getcwd())

    coeffs = ""
    if x_mask is not None and len(x_mask) > 0:
        coeffs = " " + " ".join(shlex.quote(str(v)) for v in x_mask)

    # Command executed inside the container
    inner_cmd = (
        'export PATH="$PATH:/opt/local/bin"; '
        f"cd {shlex.quote(workdir)}; "
        f"python analysis_cli.py {shlex.quote(root_file)}{coeffs}"
    )

    # Use $SINGULARITY and $SIF from init_env
    outer_cmd = f'$SINGULARITY exec "$SIF" /bin/bash -lc {shlex.quote(inner_cmd)}'

    res = subprocess.run(
        outer_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if res.returncode != 0:
        raise RuntimeError("[ERROR meas_kin_in_container] Failed inside container.")

    stdout = res.stdout.strip()
    if not stdout:
        raise RuntimeError("[ERROR meas_kin_in_container] No output.")

    last_line = stdout.splitlines()[-1]
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError as e:
        print("[ERROR meas_kin_in_container] JSON decode error:", e)
        print("[ERROR meas_kin_in_container] Full STDOUT:\n", stdout)
        raise

    return data
