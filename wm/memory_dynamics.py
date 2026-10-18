#!/usr/bin/env python3
"""Finite-memory latent world model: explicit multi-timescale decay of the
memory residual m_t, with the chemical latent split as

    z_chem(t) = z_eq(S_t, E_t) + m_t
    dry transition:  m <- m * exp(-dt/tau)      (per-channel learnable tau)
    wet transition:  m <- m * exp(-dt/tau) + g(S_t, E_t, m) * dt
    future damage:   dd = F(S_t, E_t, z_chem, dt)

Teacher (privileged, training only): geometry-aligned local near-pit chemistry
(12 features, tools/local_chem.py).  At test the model uses ONLY observable
inputs, carrying m through the trajectory.

Evaluation:
  - 3 matched pairs (A/B divergence ratios)   [memory amplitude + decay]
  - 358 dry-gap pairs: latent decay curve R(dt) = ||m_after||/||m_before||
  - learned tau per channel
  - 3 seeds, mean +- std

Usage: llm-python wm/memory_dynamics.py [--seeds 3] [--epochs 120]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "wm"))
from faraday_depth import parse_iloc, faraday, resample_arc  # noqa: E402
from local_chem import extract_local_features                 # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DSLOW, DENV, DMEM = 128, 6, 12


def env_feat(t, T, dT, cl, film):
    return np.array([T / 25.0, dT / 6.0, float(np.cos(2 * np.pi * t)), float(np.sin(2 * np.pi * t)),
                     np.log1p(cl) / 7.0, film / 10.0], np.float32)


def load_traj(prefix, T, dT, film, wets, cls):
    prefix = str(prefix)
    t, coords, iloc = parse_iloc(prefix + "_iloc.txt")
    depth, cr = faraday(iloc, t)
    _, dp = resample_arc(coords, depth)
    F = dp.shape[1]
    t = t[:F]
    wet = np.zeros(F, bool); cl = np.zeros(F, np.float32)
    for k, w in enumerate(wets):
        m = (t >= k * 1.0 - 1e-9) & (t <= k * 1.0 + w + 1e-9)
        wet |= m; cl[m] = cls[k]
    env = np.stack([env_feat(t[j], T, dT, cl[j], film) for j in range(F)])
    local = extract_local_features(prefix)[:F]
    return {"t": t, "depth": dp * 1e6, "wet": wet, "env": env, "local": local}


class MemDyn(nn.Module):
    def __init__(self, hidden=256, dmem=DMEM):
        super().__init__()
        self.zeq = nn.Sequential(nn.Linear(DSLOW + DENV, hidden), nn.SiLU(),
                                 nn.Linear(hidden, dmem))
        self.mupd = nn.Sequential(nn.Linear(DSLOW + DENV + dmem, hidden), nn.SiLU(),
                                  nn.Linear(hidden, dmem))
        self.head_net = nn.Sequential(nn.Linear(DSLOW + DENV + dmem + 1, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, DSLOW))
        self.log_tau = nn.Parameter(torch.full((dmem,), math.log(3.0 / 24.0)))

    def forward(self, slow, env, wet, dt):
        """slow (F,128), env (F,7), wet (F,), dt (F-1,) -> zchem (F,12), m (F,12)."""
        F = slow.shape[0]
        tau = self.log_tau.exp()
        m = torch.zeros(self.log_tau.shape[0], device=slow.device)
        zs, ms = [], []
        for j in range(F):
            z_eq = self.zeq(torch.cat([slow[j], env[j]]))
            zs.append(z_eq + m)
            ms.append(m)
            if j < F - 1:
                d = dt[j]
                decay = torch.exp(-d / tau)
                if wet[j] > 0.5:
                    upd = self.mupd(torch.cat([slow[j], env[j], m]))
                    m = m * decay + upd * d
                else:
                    m = m * decay
        return torch.stack(zs), torch.stack(ms)

    def predict(self, slow, env, z, dt):
        return self.head_net(torch.cat([slow, env, z, dt.unsqueeze(-1)], -1))


def make_samples(traj, depth_scale, dscale):
    """within-wet (j,k) pairs -> tensors."""
    t, dep, wet = traj["t"], traj["depth"], traj["wet"]
    F = len(t)
    jj, kk = [], []
    for j in range(F):
        if not wet[j]:
            continue
        for h in (1, 2, 3, 5, 8, 12):
            k = j + h
            if k < F and wet[k]:
                jj.append(j); kk.append(k)
    jj, kk = np.array(jj), np.array(kk)
    dt = t[kk] - t[jj]
    y = (dep[:, kk] - dep[:, jj]).T / dscale          # (N,128)
    return jj, kk, dt, y


def train_one(trajs, seeds_data, args, seed, lam=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MemDyn().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    tdata = []
    for tr in trajs:
        slow = torch.tensor((tr["depth"] / seeds_data["depth_scale"]).T, dtype=torch.float32, device=DEV)  # (F,128)
        env = torch.tensor(tr["env"], device=DEV)
        wet = torch.tensor(tr["wet"].astype(np.float32), device=DEV)
        dt = torch.tensor(np.diff(tr["t"]), dtype=torch.float32, device=DEV)
        teacher = torch.tensor(seeds_data["norm_local"](tr["local"]), device=DEV)
        jj, kk, dts, y = make_samples(tr, seeds_data["depth_scale"], seeds_data["dscale"])
        tdata.append((slow, env, wet, dt, teacher,
                      torch.tensor(jj, device=DEV), torch.tensor(kk, device=DEV),
                      torch.tensor(dts, dtype=torch.float32, device=DEV),
                      torch.tensor(y, dtype=torch.float32, device=DEV)))
    n = len(tdata)
    for ep in range(args.epochs):
        perm = np.random.permutation(n)
        tot_l = tot_f = 0.0
        for ti in perm:
            slow, env, wet, dt, teacher, jj, kk, dts, y = tdata[ti]
            z, m = model(slow, env, wet, dt)
            l_lat = ((z - teacher) ** 2).mean()
            pred = model.predict(slow[jj], env[jj], z[jj], dts)
            l_fut = ((pred - y) ** 2).mean()
            loss = l_fut + lam * l_lat
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_l += float(l_lat); tot_f += float(l_fut)
        if (ep + 1) % 20 == 0:
            print(f"  seed{seed} ep{ep+1:4d}  L_fut {tot_f/n:.4f}  L_lat {tot_l/n:.4f}  "
                  f"tau[h] {np.round(model.log_tau.exp().detach().cpu().numpy()*24, 2)}", flush=True)
    return model


def pair_predict(model, pair, meta, switch=2.0, end=2.9, depth_scale=1.0):
    tr = pair
    slow = torch.tensor((tr["depth"] / depth_scale).T, dtype=torch.float32, device=DEV)
    env = torch.tensor(tr["env"], device=DEV)
    wet = torch.tensor(tr["wet"].astype(np.float32), device=DEV)
    dt = torch.tensor(np.diff(tr["t"]), dtype=torch.float32, device=DEV)
    with torch.no_grad():
        z, m = model(slow, env, wet, dt)
        i = int(np.argmin(np.abs(tr["t"] - switch)))
        j = int(np.argmin(np.abs(tr["t"] - end)))
        dtt = torch.tensor([tr["t"][j] - tr["t"][i]], dtype=torch.float32, device=DEV)
        pred = model.predict(slow[i:i+1], env[i:i+1], z[i:i+1], dtt)[0].cpu().numpy() * meta["dscale"]
    true_inc = (tr["depth"][:, j] - tr["depth"][:, i])
    return pred, true_inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "data/chained_hist/params.csv")))
    trajs = []
    for r in rows:
        p = ROOT / "data/chained_hist/runs" / r["name"]
        if not Path(str(p) + "_iloc.txt").exists():
            continue
        trajs.append(load_traj(p, float(r["T"]), float(r["dT"]), float(r["film"]),
                               [float(x) for x in r["wets"].split(",")],
                               [float(x) for x in r["cl"].split(",")]))
    print(f"loaded {len(trajs)} trajectories")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(trajs))
    n_val = max(2, int(0.15 * len(trajs)))
    val_ids = set(idx[:n_val].tolist())
    tr = [t for i, t in enumerate(trajs) if i not in val_ids]
    va = [t for i, t in enumerate(trajs) if i in val_ids]

    all_dep = np.concatenate([t_["depth"].ravel() for t_ in tr])
    depth_scale = float(np.percentile(all_dep, 99)) + 1e-9
    all_local = np.concatenate([t_["local"] for t_ in tr], 0)
    loc_mu = all_local.mean(0); loc_sd = all_local.std(0) + 1e-6
    def norm_local(x):
        return (x - loc_mu) / loc_sd
    # increment scale
    incs = []
    for t_ in tr:
        for j, k, _, y in [make_samples(t_, depth_scale, 1.0)]:
            incs.append(y.ravel())
    dscale = float(np.concatenate(incs).std()) + 1e-9
    sd = {"depth_scale": depth_scale, "dscale": dscale, "norm_local": norm_local,
          "loc_mu": loc_mu, "loc_sd": loc_sd}
    print(f"depth_scale {depth_scale:.4f} um | increment scale {dscale:.5f} um")

    # ---- matched pairs ----
    PAIRS = {
        "gap2.4h": (2.9, {"A": ("data/histed/pS_A", [0.9, 0.9, 0.9], [100, 800, 400]),
                          "B": ("data/histed/pS_B0875", [0.875, 0.9, 0.9], [800, 100, 400])}),
        "gap4.8h": (2.9, {"A": ("data/histed/sweep2/A_g020", [0.9, 0.8, 0.9], [100, 800, 400]),
                          "B": ("data/histed/sweep2/Bm_g020", [0.8331, 0.8, 0.9], [800, 100, 400])}),
        "gap16.8h": (2.7, {"A": ("data/histed/pL_A", [0.3, 0.3, 0.7], [100, 800, 400]),
                           "B": ("data/histed/pL_B", [0.3, 0.3, 0.7], [800, 100, 400])}),
    }
    pair_trajs = {}
    for pname, (end, members) in PAIRS.items():
        pair_trajs[pname] = (end, {tag: load_traj(ROOT / pre, 25.0, 6.0, 10.0, wets, cls)
                                   for tag, (pre, wets, cls) in members.items()})

    # ---- 358 dry-gap pairs: true latent decay curve ----
    gap_pairs = []   # (dt, traj_idx, j0, j1)
    for ti, t_ in enumerate(tr):
        tt, wet = t_["t"], t_["wet"]
        F = len(tt)
        for i in range(1, F):
            if wet[i] and not wet[i - 1]:
                j = i - 1
                while j >= 0 and not wet[j]:
                    j -= 1
                if j >= 0:
                    gap_pairs.append((tt[i] - tt[j], ti, j, i))
    print(f"dry-gap pairs: {len(gap_pairs)}")

    # memory-type local features whose dry-season equilibrium is ~0:
    # fe2_pit (0), |grad fe2| (3), |grad ph| (4), iloc_pit (11).
    # Scale by q95 (NOT mean-centering) so the equilibrium maps to ~0 and the
    # norm decays with memory.
    MEM_IDX = [0, 3, 4, 11]
    mem_scale = np.percentile(all_local[:, MEM_IDX], 95, axis=0) + 1e-9

    def true_memory(t_, j):
        return t_["local"][j][MEM_IDX] / mem_scale

    BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.85)]
    true_curve = []
    for lo, hi in BINS:
        vals = [np.linalg.norm(true_memory(tr[ti], j1)) / (np.linalg.norm(true_memory(tr[ti], j0)) + 1e-12)
                for dt, ti, j0, j1 in gap_pairs if lo <= dt < hi]
        true_curve.append(float(np.median(vals)) if vals else float("nan"))
    print("true latent decay curve R_m:", [round(v, 3) for v in true_curve])

    # ---- seeds ----
    results = []
    for si in range(args.seeds):
        model = train_one(tr, sd, args, seed=si)
        # matched pairs
        ratios = {}
        for pname, (end, members) in pair_trajs.items():
            preds = {}
            for tag, trj in members.items():
                pred, true_inc = pair_predict(model, trj, sd, 2.0, end, depth_scale)
                preds[tag] = (pred, true_inc)
            tr_ratio = float(preds["A"][1].max() / preds["B"][1].max())
            pr_ratio = float(preds["A"][0].max() / preds["B"][0].max())
            ratios[pname] = (tr_ratio, pr_ratio)
        # latent decay curve (model m)
        model_curve = []
        with torch.no_grad():
            ms_by_traj = []
            for t_ in tr:
                slow = torch.tensor((t_["depth"] / depth_scale).T, dtype=torch.float32, device=DEV)
                env = torch.tensor(t_["env"], device=DEV)
                wet = torch.tensor(t_["wet"].astype(np.float32), device=DEV)
                dtt = torch.tensor(np.diff(t_["t"]), dtype=torch.float32, device=DEV)
                _, m = model(slow, env, wet, dtt)
                ms_by_traj.append(m.cpu().numpy())
        for lo, hi in BINS:
            vals = []
            for dtt_, ti, j0, j1 in gap_pairs:
                if lo <= dtt_ < hi:
                    m0 = np.linalg.norm(ms_by_traj[ti][j0]); m1 = np.linalg.norm(ms_by_traj[ti][j1])
                    if m0 > 1e-9:
                        vals.append(m1 / m0)
            model_curve.append(float(np.median(vals)) if vals else float("nan"))
        taus = np.round(model.log_tau.exp().detach().cpu().numpy() * 24, 2)
        results.append({"ratios": ratios, "curve": model_curve, "taus": taus})
        print(f"\nseed{si}: pair ratios {[(k, round(v[1],3), round(v[0],3)) for k,v in ratios.items()]}")
        print(f"        model decay curve {[round(v,3) for v in model_curve]}")
        print(f"        tau[h] {taus}", flush=True)

    print("\n=== summary over seeds ===")
    for pname in PAIRS:
        pr = np.array([r["ratios"][pname][1] for r in results])
        tr_ = results[0]["ratios"][pname][0]
        print(f"{pname:10} TRUE {tr_:.3f} | model {pr.mean():.3f} ± {pr.std():.3f}")
    curve = np.array([r["curve"] for r in results])
    print("decay curve true :", [round(v, 3) for v in true_curve])
    print("decay curve model:", [f"{m:.3f}±{s:.3f}" for m, s in zip(curve.mean(0), curve.std(0))])
    taus = np.array([r["taus"] for r in results])
    print(f"learned tau[h] mean±std: {np.round(taus.mean(0),2)} ± {np.round(taus.std(0),2)}")


if __name__ == "__main__":
    main()
