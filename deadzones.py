""" Modeling sensitive active/dead zones on B0 disks """


# librairies

import math
import numpy as np

# Constants

B0_DISKS = {
    "x0": -150.0,
    "y0": 0.0,
    "Rint": 35.0,
    "Rext": 150.0,
    "shift": -6.5,
    }

# Zernike polynomials


def zernike_radial(n, m, r):
    """
    Radial component of Zernike polynomials.

    Args:
        n (int): First Zernike index.
        m (int): Second Zernike index.
        r (int): Radius (mm).
        theta (float): Angle (rad).

    Returns:
        R (numpy array): Radial component of Zernike (n,m) at radius r.
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
    Complete Zernike polynomials.

    Args:
        n (int): First Zernike index.
        m (int): Second Zernike index.
        r (int): Radius (mm).
        theta (float): Angle (rad).

    Returns:
        float: Zernike (n,m) at (r,theta).
    """
    Rnm = zernike_radial(n, abs(m), r)
    if m >= 0:
        return Rnm * np.cos(m * theta)
    else:
        return Rnm * np.sin(abs(m) * theta)
    


# Cut Zernike basis


def zernike_basis(N, r, theta):
    """
    Zernike polynomial basis cut.

    Args:
        N (int): Order cut-off.
        r (float): Radius (mm).
        theta (float): Angle (rad).

    Returns:
        basis (dict): Basis of Zernike polynomials (index n, m)
    """
    basis = {}
    for n in range(N+1):
        for m in range(-n, n+1, 2):
            basis[(n,m)] = zernike(n, m, r, theta)
    return basis


# Zernike-based probability density function (PDF)


def zernike_pdf(c, N, r, theta):
    """
    Probability distribution from a Zernike polynomials basis.

    Args:
        c (list float): Linear combination coefficients vector.
        N (int): Order cut-off.
        r (float): Radius (mm).
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

    g0 = g - np.max(g)
    f = np.exp(g0)

    # normalization

    jacobian = r
    f_norm = f / np.sum(f * jacobian)

    # return

    return f_norm


# Deadzones mask


def zernike_mask_xy(c, N, sigma, x, y, rmax=1.0, rmin=0.0, center=(0.0, 0.0)):
    """
    Mask based on Zernike polynomials pdf with a given coverage ratio..

    Args:
        c (list, float): Linear combination coefficients vector.
        N (int): Order cut-off.
        sigma (float): Coverage ratio (0 to 1).
        x, y (array-like): Physical positions.
        rmax (float): Outer radius (mm).
        rmin (float): Inner radius (mm).
        center (tuple): Center coordinates (mm).

    Returns:
        mask (ndarray, int): binary array (0 or 1).
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


def make_zernike_mask(c, N, sigma, rmax, rmin, center):
    """
    Deadzones mask factory.

    Args:
        c (list float): Linear combination coefficients vector.
        N (int): Order cut-off.
        sigma (float): Coverage ratio (0 to 1).
        x, y (array-like): Physical positions.
        rmax (float): Outer radius (mm).
        rmin (float): Inner radius (mm).
        center (tuple): Center coordinates (mm).
    
    Returns: 
        _mask (function): Mask function.
    """

    def _mask(x, y):
        m = zernike_mask_xy(c, N, sigma, x, y, rmax=rmax, rmin=rmin, center=center)
        return m
    
    return _mask
