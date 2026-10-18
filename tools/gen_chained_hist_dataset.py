#!/usr/bin/env python3
"""Generate a diverse chained wet/dry history dataset (Step 5 training data).

Each trajectory: 4 wet periods on a 1-day cycle with per-period chloride loads
(log-uniform 80-900 mg/m^2), per-period wet durations (0.55-0.95 d), random
T_mean/T_amp/film.  Dense frames within each wet period + post-end frames.

Jobs are split into 4 chunks run by 4 parallel COMSOL JVMs (detached).

Usage: .venv/bin/python tools/gen_chained_hist_dataset.py [--n 200] [--chunks 4] [--launch]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/chained_hist"
RUNS = OUT / "runs"
MPH = ROOT / "comsol/templates/q235_env_v3.mph"
RNG = np.random.default_rng(20261017)

N_WET = 4
CYCLE = 1.0
TAU = 0.01
CL0INIT = 400


def tlist_for(wets):
    ts = []
    for i, wet in enumerate(wets):
        s = i * CYCLE
        ts += [s, s + 0.02, s + 0.05, s + 0.1]
        x = s + 0.2
        while x <= s + wet - 0.05 + 1e-9:
            ts.append(round(x, 3)); x += 0.1
        ts += [round(s + wet, 3), round(s + wet + 0.05, 3)]
        # dry-gap midpoint frame
        gap_mid = s + wet + 0.05 + (CYCLE - wet - 0.05) / 2
        if gap_mid < s + CYCLE - 0.02:
            ts.append(round(gap_mid, 3))
    return ",".join(str(t) for t in sorted(set(round(t, 3) for t in ts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(args.n):
        name = f"h{k:04d}"
        wet = np.round(RNG.uniform(0.55, 0.95, N_WET), 3)
        cl = np.exp(RNG.uniform(np.log(80), np.log(900), N_WET)).round(1)
        T = round(float(RNG.uniform(18, 27)), 2)
        dT = round(float(RNG.uniform(3, 8)), 2)
        film = float(RNG.choice([5, 10, 20]))
        rows.append({
            "name": name, "T": T, "dT": dT, "film": film,
            "wets": ",".join(str(w) for w in wet),
            "cl": ",".join(str(c) for c in cl),
            "tlist": tlist_for(wet),
            "out": str(RUNS / name),
        })

    # params table
    with open(OUT / "params.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # chunked jobs files
    chunks = np.array_split(np.arange(len(rows)), args.chunks)
    files = []
    for ci, idx in enumerate(chunks):
        p = OUT / f"jobs_c{ci}.csv"
        with open(p, "w") as f:
            f.write("# outprefix;T;film;Tamp;n_wet;wet_d;cyc_d;tau;cl_list;tlist;cl0_init\n")
            for i in idx:
                r = rows[i]
                f.write(f"{r['out']};{r['T']};{r['film']};{r['dT']};{N_WET};{r['wets']};"
                        f"{CYCLE};{TAU};{r['cl']};{r['tlist']};{CL0INIT}\n")
        files.append(p)
        print(f"wrote {p} ({len(idx)} jobs)")

    if args.launch:
        if not MPH.exists():
            raise FileNotFoundError(f"COMSOL model template required: {MPH} (not distributed)")
        procs = []
        for ci, p in enumerate(files):
            log = open(OUT / f"run_c{ci}.log", "w")
            proc = subprocess.Popen(
                ["bash", "tools/jrun.sh", "ChainedRunner2", str(MPH), str(p)],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            procs.append((ci, proc.pid))
            print(f"launched chunk {ci}: pid {proc.pid}")
        (OUT / "pids.txt").write_text("\n".join(f"{ci} {pid}" for ci, pid in procs))


if __name__ == "__main__":
    main()
