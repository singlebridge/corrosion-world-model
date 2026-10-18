#!/usr/bin/env python3
"""Faraday-integrate boundary i_loc exports into B-group profiles (spec 2).

depth(x, t_k) = sum_{j<k} i_loc(x, t_j) * M_Fe/(z F rho_Fe) * dt_j   [m]
CR(x, t_k)    = i_loc(x, t_k) * M_Fe/(z F rho_Fe)                   [mm/a]

Profiles are resampled onto 128 bins along the boundary arc length.

Usage: .venv/bin/python tools/faraday_depth.py <prefix>   (prefix_iloc.txt etc.)
Writes <prefix>_B.npz with depth/cr arrays + arc coordinates.
"""
import sys
from pathlib import Path

import numpy as np

M_FE = 55.8e-3      # kg/mol
Z_FE = 2.0          # electrons
F_CONST = 96485.0   # C/mol
RHO_FE = 7860.0     # kg/m^3
V_PER_AMP = M_FE / (Z_FE * F_CONST * RHO_FE)      # m/s per (A/m^2)
S_PER_YEAR = 365.25 * 86400.0
NBINS = 128


def parse_iloc(path):
    header, rows = None, []
    for ln in open(path):
        if ln.startswith("%"):
            if "@ t=" in ln:
                header = ln
            continue
        if ln.strip():
            rows.append([float(x) for x in ln.split()])
    times_days = [float(t) for t in __import__("re").findall(r"t=([0-9eE.+-]+)", header)]
    data = np.array(rows)
    coords = data[:, :2]
    iloc = data[:, 2:]                     # (Nnodes, nframes)
    assert len(times_days) == iloc.shape[1], (len(times_days), iloc.shape)
    return np.array(times_days), coords, iloc


def faraday(depth_v, times_days):
    """depth_v: (N, nframes) i_loc; -> depth (N, nframes) in meters; CR in mm/a."""
    v = depth_v * V_PER_AMP                          # m/s
    dt_s = np.diff(times_days) * 86400.0
    depth = np.zeros_like(v)
    for k in range(1, v.shape[1]):
        depth[:, k] = depth[:, k - 1] + v[:, k - 1] * dt_s[k - 1]
    cr = v * 1e3 * S_PER_YEAR                        # mm/a
    return depth, cr


def resample_arc(coords, arr, nbins=NBINS):
    """Resample (N, F) along cumulative arc length of (N,2) coords -> (nbins, F)."""
    d = np.sqrt(((np.diff(coords, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    su = np.linspace(0.0, s[-1], nbins)
    order = np.argsort(s)
    out = np.empty((nbins, arr.shape[1]))
    for k in range(arr.shape[1]):
        out[:, k] = np.interp(su, s[order], arr[order, k])
    return su, out


def main():
    prefix = Path(sys.argv[1] if len(sys.argv) > 1 else
                  "data/pilot/full0")
    times_days, coords, iloc = parse_iloc(f"{prefix}_iloc.txt")
    depth, cr = faraday(iloc, times_days)
    su, depth_p = resample_arc(coords, depth)
    _, cr_p = resample_arc(coords, cr)

    np.savez(f"{prefix}_B.npz", times_days=times_days, arc_m=su,
             depth_m=depth_p, cr_mm_a=cr_p, coords=coords, iloc=iloc)

    print(f"frames: {len(times_days)} (days: {list(times_days)})")
    print(f"boundary nodes: {coords.shape[0]}, arc length: {su[-1]*1e6:.3f} um")
    print(f"i_loc range: [{iloc.min():.3e}, {iloc.max():.3e}] A/m^2")
    print(f"CR range:    [{cr.min():.3e}, {cr.max():.3e}] mm/a")
    print(f"depth @last: max {depth[:, -1].max()*1e9:.3f} nm  (mean {depth[:, -1].mean()*1e9:.3f} nm)")
    print(f"depth @last as um: {depth[:, -1].max()*1e6:.6f} um")
    print(f"wrote {prefix}_B.npz")


if __name__ == "__main__":
    main()
