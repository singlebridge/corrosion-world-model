#!/usr/bin/env python3
"""Geometry-aligned LOCAL near-pit chemistry features (the oracle's privileged state).

For each frame, locate the pit bottom (max accumulated depth on the electrode
boundary) and extract, at that geometric location:
  - Fe2+, pH, potential value at the nearest domain node
  - |grad Fe2+|, |grad pH| from a local plane fit
  - Fe2+/pH/potential means within 0.5 um and 1.5 um of the pit
  - the local dissolution current density i_loc at the pit node

Returns (F, 12).  Requires <prefix>_{fe2plus,ph,potential,iloc}.txt.
"""
import re

import numpy as np

RADII = (0.5e-6, 1.5e-6)
N_FEAT = 12


def parse_export(path):
    times = None
    for ln in open(path):
        if not ln.startswith("%"):
            break
        if "@ t=" in ln:
            times = [float(t) for t in re.findall(r"t=([0-9eE.+-]+)", ln)]
    dat = np.loadtxt(path, comments="%")
    return (np.array(times) if times is not None else None), dat[:, :2], dat[:, 2:]


def plane_grad(x, y, v):
    """Least-squares plane fit -> |grad|."""
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return float(np.hypot(coef[0], coef[1]))


def extract_global_stats(prefix):
    """Global per-frame stats (12): [fe2 max/mean/std/p90, ph max/mean/std/p90,
    pot max/mean/std/p90] -- the reference 'global oracle' that failed."""
    out = []
    for ch in ("fe2plus", "ph", "potential"):
        _, _, v = parse_export(str(prefix) + f"_{ch}.txt")
        out.append(np.stack([v.max(0), v.mean(0), v.std(0), np.percentile(v, 90, axis=0)], axis=1))
    return np.concatenate(out, axis=1).astype(np.float32)


def extract_features(prefix):
    """(local (F,12), global (F,12)) -- one parse of the channel exports."""
    return extract_local_features(prefix), extract_global_stats(prefix)


def extract_local_features(prefix):
    from faraday_depth import faraday
    prefix = str(prefix)
    _, dcoords, fe2 = parse_export(prefix + "_fe2plus.txt")
    _, _, ph = parse_export(prefix + "_ph.txt")
    _, _, pot = parse_export(prefix + "_potential.txt")
    times, bcoords, iloc = parse_export(prefix + "_iloc.txt")
    depth_v, _ = faraday(iloc, times)          # per-boundary-node accumulated depth

    F = min(fe2.shape[1], ph.shape[1], pot.shape[1], iloc.shape[1], depth_v.shape[1])
    feats = np.zeros((F, N_FEAT), np.float32)
    for k in range(F):
        pit = int(np.argmax(depth_v[:, k]))
        pc = bcoords[pit]
        d = np.linalg.norm(dcoords - pc, axis=1)
        near = int(np.argmin(d))
        feats[k, 0] = fe2[near, k]
        feats[k, 1] = ph[near, k]
        feats[k, 2] = pot[near, k]
        m = d <= RADII[1]
        if m.sum() >= 3:
            feats[k, 3] = plane_grad(dcoords[m, 0] - pc[0], dcoords[m, 1] - pc[1], fe2[m, k])
            feats[k, 4] = plane_grad(dcoords[m, 0] - pc[0], dcoords[m, 1] - pc[1], ph[m, k])
        idx = 5
        for r in RADII:
            mm = d <= r
            if mm.sum() == 0:
                mm = d <= d.min() * 1.01
            feats[k, idx] = fe2[mm, k].mean()
            feats[k, idx + 1] = ph[mm, k].mean()
            feats[k, idx + 2] = pot[mm, k].mean()
            idx += 3
        feats[k, 11] = iloc[pit, k]
    return feats


if __name__ == "__main__":
    import sys
    f = extract_local_features(sys.argv[1])
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print("shape", f.shape)
    print("first frame:", f[0])
    print("last frame :", f[-1])
