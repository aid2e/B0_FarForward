""" Analysis of ROOT files from simulations """

import sys
import json
from analysis import meas_kin
from deadzones import B0_DISKS, make_zernike_mask

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python analysis_cli.py <root_file> [mask_coeffs...]", file=sys.stderr)
        sys.exit(1)

    root_file = sys.argv[1]
    x_mask = [float(i) for i in sys.argv[2:]]

    mask = None
    if len(x_mask) > 0:
        mask = make_zernike_mask(
            x_mask,
            N=2,  # degree cutoff
            rmax=B0_DISKS["Rext"],
            rmin=B0_DISKS["Rint"],
            center=(B0_DISKS["x0"], B0_DISKS["y0"]),
            sigma=0.8,  # coverage ratio
        )

    res = meas_kin(root_file, mask=mask)
    print(json.dumps(res))