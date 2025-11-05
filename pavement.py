""" Modelling of silicon detectors pavements on B0 disks """


# librairies


import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib


# B0 geom

B0_PLANES = {
    "tracker1": (5800.0, 6000.0),
    "tracker2": (6100.0, 6300.0),
    "tracker3": (6400.0, 6500.0),
    "tracker4": (6600.0, 6800.0),
}

B0_DISKS = {
    "x0": -150.0,
    "y0": 0.0,
    "Rint": 35.0,
    "Rext": 150.0,
    "shift": -6.5,
}

# path


FIG_PATH = Path("/volatile/eic/fraisse/mobo/fig/")


# Zerkine radial component


def zernike_radial(n, m, r):
    """
    Radial component of Zerkine polynomials.

    Args:
        n (int): First Zerkine index.
        m (int): Second Zerkine index.
        r (int): Radius.
        theta (float): Angle (rad).

    Returns:
        R (numpy array): Radial component of Zerkine (n,m) at radius r.
    """

    # n-m odd

    if (n - m) % 2 != 0:
        return np.zeros_like(r)
    
    # n-m even
    
    R = np.zeros_like(r)
    for k in range((n - m)//2 + 1):
        coeff = (pow(-1,k) * math.factorial(n - k) /
                 (math.factorial(k) *
                  math.factorial((n + m)//2 - k) *
                  math.factorial((n - m)//2 - k)))
        R += coeff * pow(r,n - 2*k)
    return R
    
def zernike(n, m, r, theta):
    """
    Complete Zerkine polynomials.

    Args:
        n (int): First Zerkine index.
        m (int): Second Zerkine index.
        r (int): Radius.
        theta (float): Angle (rad).

    Returns:
        float: Zerkine (n,m) at (r,theta).
    """
    Rnm = zernike_radial(n, abs(m), r)
    if m >= 0:
        return Rnm * np.cos(m * theta)
    else:
        return Rnm * np.sin(abs(m) * theta)
    

# Cut Zerkine basis


def zernike_basis(N, r, theta):
    """Zerkine polynomial basis cut.

    Args:
        N (int): Maximmum order of Zerkine decomposition.
        r (float): Radius.
        theta (float): Angle (rad).

    Returns:
        list float: Basis of Zerkine polynomials 
    """
    basis = {}
    for n in range(N+1):
        for m in range(-n, n+1, 2):  # m de -n à n, même parité
            basis[(n,m)] = zernike(n, m, r, theta)
    return basis


# probability distribution


def zernike_pdf(c, N, r, theta):
    """
    Probability distribution from a linear combination of Zerkine polynomials.

    Args:
        c (list float): Coefficients of linear combination.
        N (int): Maximum order.
        r (float): Radius.
        theta (float): Angle (rad).

    Returns:
        f_norm (numpy array): Probability distribution. 
    """

    # linear combination

    basis = zernike_basis(N, r, theta)
    keys = sorted(basis.keys(), key=lambda x: (x[0], x[1]))
    g = np.zeros_like(r)
    for coeff, key in zip(c, keys):
        g += coeff * basis[key]

    # log-density probability

    f = np.exp(g)

    # normalization

    jacobian = r
    f_norm = f / np.sum(f * jacobian)

    # return

    return f_norm


# mask


def zernike_mask_xy(c, N, sigma, x, y, rmax=1.0, rmin=0.0, center=(0.0, 0.0)):
    """
    Mask based on Zernike polynomials pdf with a given coverage ratio..

    Args:
        c (list, float): Linear combination coefficients.
        N (int): Maximum order.
        sigma (float): Coverage ratio (0..1).
        x, y (array-like): Physical positions.
        rmax (float): Outer radius.
        rmin (float): Inner radius (hole).
        center (tuple): (cx, cy).

    Returns:
        mask (ndarray, float): 0/1 mask array (same shape as x,y).
    """

    cx, cy = center
    xn = (np.asarray(x) - cx) / rmax
    yn = (np.asarray(y) - cy) / rmax
    rn = np.sqrt(xn**2 + yn**2)
    th = np.arctan2(yn, xn)

    # pdf 

    pdf = zernike_pdf(c, N, rn, th)

    # disks surface

    rmin_rel = (rmin / rmax) if rmax > 0 else 0.0
    pdf[(rn > 1.0) | (rn < rmin_rel)] = 0.0

    # sorting

    values = pdf.ravel()
    jacobian = rn.ravel()
    weights = values * jacobian

    if np.all(values == 0):
        return np.zeros_like(pdf, dtype=float)

    idx = np.argsort(values)[::-1]
    values_sorted = values[idx]
    weights_sorted = weights[idx]

    # cut-off 

    cum_weights = np.cumsum(weights_sorted)
    total = cum_weights[-1]
    if total == 0:
        return np.zeros_like(pdf, dtype=float)

    cum_frac = cum_weights / total
    sel = cum_frac <= float(sigma)
    cutoff_value = values_sorted[sel][-1] if np.any(sel) else values_sorted[0]

    # final mask

    mask = (pdf >= cutoff_value).astype(float)
    return mask


# plot triangular map


def map_triangle(N=3, res=200):
    """
    Plot Zerkine polynomials.

    Args:
        N (int, optional): Maximum order. Defaults to 3.
        res (int, optional): Mesh precision. Defaults to 200.

    Returns:
        fig_path (str): PNG file path. 
    """

    # mesh

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # figure design

    fig, axes = plt.subplots(N+1, N+1, figsize=(2*(N+1), 2*(N+1)))
    for ax in axes.flat:
        ax.axis("off")

    # plots

    for n in range(N+1):
        ms = list(range(-n, n+1, 2))
        for j, m in enumerate(ms):
            ax = axes[n, j] 
            Z = np.zeros_like(R)
            Z[R <= 1] = zernike(n, m, R[R <= 1], Theta[R <= 1])
            im = ax.imshow(Z, extent=(-1,1,-1,1), origin="lower",
                           cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_title(f"n={n}, m={m}", fontsize=8)
            plt.gca().set_frame_on(False)
            ax.set_xticks([]); 
            ax.set_yticks([])

    # save & return

    final_path = FIG_PATH / "zerkine.png"
    plt.savefig(final_path)
    return final_path


# plot probability map


def map_pdf(c, N=2, res=1000):
    """
    Probability map plot.

    Args:
        c (list, float): Linear combination coefficients.
        N (int, optional): Maximum order. Defaults to 3.
        res (int, optional): Mesh precision. Defaults to 200.

    Returns:
        fig_path (str): PNG file path. 
    """

    # mesh

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # plots

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    Z = np.zeros_like(R)
    Z[R <= 1] = zernike_pdf(c, N, R[R <= 1], Theta[R <= 1])
    im = ax.imshow(Z, origin="lower", cmap="coolwarm")
    plt.gca().set_frame_on(False)
    #ax.set_xticks([])
    #ax.set_yticks([])

    # save & return

    final_path = FIG_PATH / "zerkine_pdf.png"
    plt.savefig(final_path)
    return final_path


# plot mask map


def map_mask(c, N=2, res=200, rmax=100.0, rmin=10.0, center=(0.0, 0.0), sigma=0.8):
    """
    Plot mask.

    Args:
        c (list, float): Linear combination coefficients.
        N (int, optional): Maximum order. Defaults to 3.
        res (int, optional): Mesh precision. Defaults to 200.
        rmax (float): B0 disk radius.
        rmin (float): B0 disk internal annulus.
        center (couple float): B0 disk center.
        sigma (float): Active zone ratio of B0 disks.

    Returns:
        fig_path (str): PNG file path. 
    """
    cx, cy = center

    xmin, xmax = cx - rmax, cx + rmax
    ymin, ymax = cy - rmax, cy + rmax

    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(x, y)

    # --- calcul du masque ---
    Z = zernike_mask_xy(c, N, sigma, X, Y, rmax=rmax, rmin=rmin, center=center)

    # --- forcer le blanc hors de l’anneau ---
    Rphys = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    Z = np.where((Rphys >= rmin) & (Rphys <= rmax), Z, np.nan)

    # --- plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap_m = matplotlib.colormaps["coolwarm"].copy()
    cmap_m.set_bad(color="white")  # NaN → blanc
    im = ax.imshow(
        Z, origin="lower", extent=(xmin, xmax, ymin, ymax),
        vmin=0.0, vmax=1.0, cmap=cmap_m, interpolation="nearest"
    )
    ax.set_aspect("equal")
    ax.set_frame_on(False)

    out_path = FIG_PATH / "zernike_mask.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return out_path


# mask for simulated data




# test


# if __name__ == "__main__":

#     # map_triangle()
#     # c = [
#     # 0.0, 0.0, 0.0,    0.0, -0.30, 0.50,
#     # 0.0, 0.0, 0.0,    0.0,  0.0,  0.0,
#     # 0.0,  1.0,  2.0,  0.0,  0.0, 0.0,
#     # 0.0,  0.0,  0.0
#     # ]

#     # map_pdf(c,N=5)
#     # map_mask(c,N=5, res=200, rmax=B0_DISKS["Rext"], rmin=B0_DISKS["Rint"], center=(B0_DISKS["x0"], B0_DISKS["y0"]), sigma=0.8)

#     c = [0.0, 0.0, 0.0, 1.0, 1.0]
#     map_mask(c,N=2, res=200, rmax=B0_DISKS["Rext"], rmin=B0_DISKS["Rint"], center=(B0_DISKS["x0"], B0_DISKS["y0"]), sigma=0.8)