#!/usr/bin/env python3
"""Stabilization study for the finite-memory latent world model.

Structure is LOCKED:
    z_chem(t) = z_eq(S_t,E_t) + m_t
    dry:  m <- m * exp(-dt/tau)          (per-channel learnable tau)
    wet:  m <- m * exp(-dt/tau) + g(S,E,m)*dt
    dd = F(S,E,z_chem,dt)

Only the tau INITIALIZATION is varied (all-3h / log-spread 1-6h / physical
spectrum), selected on the VALIDATION loss -- never on the matched pairs.

Batched (padded) training for speed.  5 seeds for the selected variant.

Usage: llm-python wm/memory_dynamics_batched.py --epochs 600 --variants 3 --seeds 3 --final-seeds 5
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
from faraday_depth import parse_iloc, faraday, resample_arc  # noqa: E402
from local_chem import extract_local_features                 # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DSLOW, DENV, DMEM = 128, 6, 12
HORIZONS = (1, 2, 3, 5, 8, 12)


def env_feat(t, T, dT, cl, film):
    return np.array([T / 25.0, dT / 6.0, float(np.cos(2 * np.pi * t)), float(np.sin(2 * np.pi * t)),
                     np.log1p(cl) / 7.0, film / 10.0], np.float32)


def load_traj(prefix, T, dT, film, wets, cls):
    prefix = str(prefix)
    t, coords, iloc = parse_iloc(prefix + "_iloc.txt")
    depth, cr = faraday(iloc, t)
    _, dp = resample_arc(coords, depth)
    F = dp.shape[1]; t = t[:F]
    wet = np.zeros(F, bool); cl = np.zeros(F, np.float32)
    for k, w in enumerate(wets):
        m = (t >= k * 1.0 - 1e-9) & (t <= k * 1.0 + w + 1e-9)
        wet |= m; cl[m] = cls[k]
    env = np.stack([env_feat(t[j], T, dT, cl[j], film) for j in range(F)])
    local = extract_local_features(prefix)[:F]
    return {"t": t, "depth": dp * 1e6, "wet": wet, "env": env, "local": local}


def pad(trajs, depth_scale, norm_local):
    Fmax = max(len(t["t"]) for t in trajs)
    B = len(trajs)
    out = {
        "slow": np.zeros((B, Fmax, DSLOW), np.float32),
        "env": np.zeros((B, Fmax, DENV), np.float32),
        "wet": np.zeros((B, Fmax), np.float32),
        "valid": np.zeros((B, Fmax), bool),
        "t": np.zeros((B, Fmax), np.float32),
        "teacher": np.zeros((B, Fmax, DMEM), np.float32),
    }
    for b, tr in enumerate(trajs):
        F = len(tr["t"])
        out["slow"][b, :F] = (tr["depth"] / depth_scale).T
        out["env"][b, :F] = tr["env"]
        out["wet"][b, :F] = tr["wet"]
        out["valid"][b, :F] = True
        out["t"][b, :F] = tr["t"]
        out["teacher"][b, :F] = norm_local(tr["local"])
    out["dt"] = np.zeros((B, Fmax - 1), np.float32)
    out["dt"][:, :] = np.clip(np.diff(out["t"], axis=1), 0.0, None)  # padding boundary dt<0 -> 0
    # per-frame extras: dmax, remaining-wet horizon (dt, y), long-gap extinction flag
    out["dmax"] = out["slow"].max(-1)                       # normalized profile max
    out["dt_rem"] = np.zeros((B, Fmax), np.float32)
    out["y_rem"] = np.zeros((B, Fmax, DSLOW), np.float32)
    out["ext"] = np.zeros((B, Fmax), np.float32)
    for b, tr in enumerate(trajs):
        F = len(tr["t"]); wet = tr["wet"]; tt = tr["t"]
        dep = tr["depth"] / depth_scale
        for j in range(F):
            if not wet[j]:
                continue
            k = j
            while k + 1 < F and wet[k + 1]:
                k += 1
            out["dt_rem"][b, j] = tt[k] - tt[j]
            out["y_rem"][b, j] = (dep[:, k] - dep[:, j]) / 1.0
            # first wet frame after a dry gap >= 6 h
            if j > 0 and not wet[j - 1]:
                m0 = j - 1
                while m0 > 0 and not wet[m0]:
                    m0 -= 1
                if m0 >= 0 and (tt[j] - tt[m0]) >= 0.25:
                    out["ext"][b, j] = 1.0
    return out


class MemDyn(nn.Module):
    def __init__(self, hidden=256, dmem=DMEM, tau_init=None):
        super().__init__()
        self.zeq = nn.Sequential(nn.Linear(DSLOW + DENV, hidden), nn.SiLU(),
                                 nn.Linear(hidden, dmem))
        self.mupd = nn.Sequential(nn.Linear(DSLOW + DENV + dmem, hidden), nn.SiLU(),
                                  nn.Linear(hidden, dmem))
        self.head_net = nn.Sequential(nn.Linear(DSLOW + DENV + dmem + 1, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, DSLOW))
        if tau_init is None:
            tau_init = torch.full((dmem,), 3.0 / 24.0)
        self.log_tau = nn.Parameter(torch.log(torch.as_tensor(tau_init, dtype=torch.float32)))

    def forward(self, slow, env, wet, dt):
        B, F, _ = slow.shape
        tau = self.log_tau.exp()
        m = torch.zeros(B, self.log_tau.shape[0], device=slow.device)
        zs, ms = [], []
        for j in range(F):
            z_eq = self.zeq(torch.cat([slow[:, j], env[:, j]], -1))
            zs.append(z_eq + m)
            ms.append(m)
            if j < F - 1:
                d = dt[:, j:j + 1].clamp(min=0.0)
                decay = torch.exp(-d / tau)
                upd = self.mupd(torch.cat([slow[:, j], env[:, j], m], -1))
                m = m * decay + wet[:, j:j + 1] * upd * d
        return torch.stack(zs, 1), torch.stack(ms, 1)

    def predict(self, slow, env, z, dt):
        return self.head_net(torch.cat([slow, env, z, dt.unsqueeze(-1)], -1))


def losses(model, batch, T, inc_scale, lam=1.0, cf_w=0.0, ext_w=0.0, bal_w=0.0):
    slow = T(batch["slow"]); env = T(batch["env"]); wet = T(batch["wet"])
    dt = T(batch["dt"]); valid = T(batch["valid"]).float(); teacher = T(batch["teacher"])
    t = T(batch["t"])
    dmax = T(batch["dmax"]); dt_rem = T(batch["dt_rem"]); ext = T(batch["ext"])
    z, m = model(slow, env, wet, dt)
    l_lat = (((z - teacher) ** 2).sum(-1) * valid).sum() / (valid.sum() * DMEM + 1e-9)
    # memory-extinction regularizer: m must vanish at first wet frames after long gaps
    if ext_w > 0 and float(ext.sum()) > 0:
        l_ext = ((m ** 2).sum(-1) * ext).sum() / (ext.sum() + 1e-9)
    else:
        l_ext = torch.zeros((), device=slow.device)
    # balanced memory weights: up-weight frames whose teacher latent is large
    mnorm = teacher.norm(dim=-1)
    wmem = 1.0 + bal_w * mnorm / (mnorm.mean() + 1e-9)
    # future increments for several horizons (within-wet, same period)
    l_fut_num = 0.0; n = 0.0
    F = slow.shape[1]
    for h in HORIZONS:
        j = torch.arange(F - h, device=slow.device)
        v = valid[:, :F - h] * valid[:, h:] * wet[:, :F - h] * wet[:, h:]
        if float(v.sum()) == 0:
            continue
        dtt = t[:, j + h] - t[:, j]
        pred = model.predict(slow[:, :F - h], env[:, :F - h], z[:, :F - h], dtt)
        y = (slow[:, h:] - slow[:, :F - h]) / inc_scale        # normalized by depth_scale; inc_scale = dscale/depth_scale
        w = v * (1.0 + bal_w * mnorm[:, :F - h] / (mnorm.mean() + 1e-9))
        l_fut_num = l_fut_num + (((pred - y) ** 2).mean(-1) * w).sum()
        n = n + v.sum()
    l_fut = l_fut_num / (n + 1e-9)
    # counterfactual pair loss: similar current damage, different hidden state
    l_cf = torch.zeros((), device=slow.device)
    if cf_w > 0:
        m_valid = valid * wet
        flat = m_valid.reshape(-1)
        idx = torch.nonzero(flat, as_tuple=False).flatten()
        if len(idx) > 400:
            idx = idx[torch.randperm(len(idx), device=slow.device)[:400]]
        bb = idx // F; jj = idx % F
        dmax_s = dmax.reshape(-1)[idx]; lat_s = teacher.reshape(-1, DMEM)[idx]
        Dd = (dmax_s[:, None] - dmax_s[None, :]).abs()
        Dl = (lat_s[:, None, :] - lat_s[None, :, :]).norm(dim=-1)
        iu = torch.triu_indices(len(idx), len(idx), offset=1, device=slow.device)
        dd = Dd[iu[0], iu[1]]; dl = Dl[iu[0], iu[1]]
        if len(dd) > 0:
            thr_d = torch.quantile(dd, 0.5); thr_l = torch.quantile(dl, 0.75)
            sel = (dd <= thr_d) & (dl >= thr_l)
            if float(sel.sum()) >= 8:
                ia, ja = iu[0][sel], iu[1][sel]
                if len(ia) > 128:
                    keep = torch.randperm(len(ia), device=slow.device)[:128]
                    ia, ja = ia[keep], ja[keep]
                ba, ja_f = bb[ia], jj[ia]; bbx, jb_f = bb[ja], jj[ja]
                dtt_a = dt_rem[bb_a := ba, ja_f]; dtt_b = dt_rem[bbx, jb_f]
                pa = model.predict(slow[ba, ja_f], env[ba, ja_f], z[ba, ja_f], dtt_a)
                pb = model.predict(slow[bbx, jb_f], env[bbx, jb_f], z[bbx, jb_f], dtt_b)
                ya = torch.tensor(batch["y_rem"][ba.cpu().numpy(), ja_f.cpu().numpy()],
                                  device=slow.device) / inc_scale
                yb = torch.tensor(batch["y_rem"][bbx.cpu().numpy(), jb_f.cpu().numpy()],
                                  device=slow.device) / inc_scale
                l_cf = (((pa - pb) - (ya - yb)) ** 2).mean()
    return l_fut, l_lat, l_ext, l_cf


def tau_init_for(kind, rng):
    if kind == "all3h":
        return torch.full((DMEM,), 3.0 / 24.0)
    if kind == "logspread":
        lh = rng.uniform(np.log(1.0), np.log(6.0), DMEM)
        return torch.tensor(np.exp(lh) / 24.0, dtype=torch.float32)
    if kind == "spectrum":
        spec = np.array([1.8, 2.3, 2.7, 3.3, 4.7])
        vals = np.resize(spec, DMEM) / 24.0
        return torch.tensor(vals, dtype=torch.float32)
    raise ValueError(kind)


def train_seed(tr_b, va_b, inc_scale, args, kind, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = MemDyn(tau_init=tau_init_for(kind, rng)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    B = tr_b["slow"].shape[0]
    best = (1e9, None)
    for ep in range(args.epochs):
        perm = rng.permutation(B)
        model.train()
        for i in range(0, B, args.bs):
            idx = perm[i:i + args.bs]
            sub = {k: v[idx] for k, v in tr_b.items()}
            lf, ll, le, lc = losses(model, sub, lambda x: torch.tensor(x, device=DEV), inc_scale,
                                    cf_w=args.cf, ext_w=args.ext, bal_w=args.bal)
            loss = lf + args.lam * ll + args.ext * le + args.cf * lc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vf, vl, ve, vc = losses(model, va_b, lambda x: torch.tensor(x, device=DEV), inc_scale,
                                    cf_w=args.cf, ext_w=args.ext, bal_w=args.bal)
        v = float(vf) + args.lam * float(vl) + args.ext * float(ve) + args.cf * float(vc)
        if v < best[0]:
            best = (v, {k: vv.clone() for k, vv in model.state_dict().items()}, (float(vf), float(vl)))
        if (ep + 1) % 100 == 0:
            print(f"    {kind} seed{seed} ep{ep+1:4d} val {v:.4f} (fut {float(vf):.4f} lat {float(vl):.4f})"
                  f" tau[h] {np.round(model.log_tau.exp().detach().cpu().numpy()*24,2)}", flush=True)
    model.load_state_dict(best[1])
    return model, best[2], best[0]


def eval_pairs(model, pair_trajs, depth_scale, dscale):
    out = {}
    for pname, (end, members) in pair_trajs.items():
        preds = {}
        for tag, trj in members.items():
            b = pad([trj], depth_scale, lambda x: x)   # norm_local unused here (teacher ignored)
            slow = torch.tensor(b["slow"], device=DEV); env = torch.tensor(b["env"], device=DEV)
            wet = torch.tensor(b["wet"], device=DEV); dt = torch.tensor(b["dt"], device=DEV)
            with torch.no_grad():
                z, m = model(slow, env, wet, dt)
                i = int(np.argmin(np.abs(trj["t"] - 2.0)))
                j = int(np.argmin(np.abs(trj["t"] - end)))
                dtt = torch.tensor([trj["t"][j] - trj["t"][i]], dtype=torch.float32, device=DEV)
                pred = model.predict(slow[:, i], env[:, i], z[:, i], dtt)[0].cpu().numpy() * dscale
            true_inc = trj["depth"][:, j] - trj["depth"][:, i]
            preds[tag] = (pred, true_inc)
        out[pname] = (float(preds["A"][1].max() / preds["B"][1].max()),
                      float(preds["A"][0].max() / preds["B"][0].max()))
    return out


def model_decay_curve(model, tr_b, gap_pairs):
    BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.85)]
    slow = torch.tensor(tr_b["slow"], device=DEV); env = torch.tensor(tr_b["env"], device=DEV)
    wet = torch.tensor(tr_b["wet"], device=DEV); dt = torch.tensor(tr_b["dt"], device=DEV)
    with torch.no_grad():
        _, m = model(slow, env, wet, dt)
    mn = m.cpu().numpy()
    curve = []
    for lo, hi in BINS:
        vals = []
        for dtv, ti, j0, j1 in gap_pairs:
            if lo <= dtv < hi:
                n0 = np.linalg.norm(mn[ti, j0]); n1 = np.linalg.norm(mn[ti, j1])
                if n0 > 1e-9:
                    vals.append(n1 / n0)
        curve.append(float(np.median(vals)) if vals else float("nan"))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)          # per variant
    ap.add_argument("--final-seeds", type=int, default=5)    # for the selected variant
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--only", type=str, default="", help="run only this tau-init variant")
    ap.add_argument("--cf", type=float, default=1.0, help="counterfactual pair loss weight")
    ap.add_argument("--ext", type=float, default=0.1, help="memory-extinction regularizer weight")
    ap.add_argument("--bal", type=float, default=1.0, help="balanced memory sampling strength")
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
    norm_local = lambda x: (x - loc_mu) / loc_sd
    dscale = 0.03728   # from the previous run (increment std, um); recompute-free
    inc_scale = dscale / depth_scale
    print(f"depth_scale {depth_scale:.4f} | dscale {dscale:.5f} | inc_scale {inc_scale:.6f}")

    tr_b = pad(tr, depth_scale, norm_local)
    va_b = pad(va, depth_scale, norm_local)

    # ---- 358 dry-gap pairs + true latent decay curve ----
    MEM_IDX = [0, 3, 4, 11]
    all_local_tr = np.concatenate([t_["local"] for t_ in tr], 0)
    mem_scale = np.percentile(all_local_tr[:, MEM_IDX], 95, axis=0) + 1e-9
    gap_pairs = []
    for ti, t_ in enumerate(tr):
        tt, wet = t_["t"], t_["wet"]
        for i in range(1, len(tt)):
            if wet[i] and not wet[i - 1]:
                j = i - 1
                while j >= 0 and not wet[j]:
                    j -= 1
                if j >= 0:
                    gap_pairs.append((tt[i] - tt[j], ti, j, i))
    BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.85)]
    true_curve = []
    for lo, hi in BINS:
        vals = []
        for dtv, ti, j0, j1 in gap_pairs:
            if lo <= dtv < hi:
                v0 = tr[ti]["local"][j0][MEM_IDX] / mem_scale
                v1 = tr[ti]["local"][j1][MEM_IDX] / mem_scale
                vals.append(np.linalg.norm(v1) / (np.linalg.norm(v0) + 1e-12))
        true_curve.append(float(np.median(vals)) if vals else float("nan"))
    print(f"dry-gap pairs {len(gap_pairs)} | true decay curve {[round(v,3) for v in true_curve]}")

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

    variants = ["all3h", "logspread", "spectrum"][:args.variants]
    if args.only:
        variants = [args.only]
    results = {}
    for kind in variants:
        results[kind] = []
        for si in range(args.seeds):
            model, (vf, vl), v = train_seed(tr_b, va_b, inc_scale, args, kind, si)
            pr = eval_pairs(model, pair_trajs, depth_scale, dscale)
            taus = np.round(model.log_tau.exp().detach().cpu().numpy() * 24, 2)
            curve = model_decay_curve(model, tr_b, gap_pairs)
            results[kind].append({"val": v, "vf": vf, "vl": vl, "pairs": pr, "taus": taus, "curve": curve})
            print(f"  {kind} seed{si}: val {v:.4f} | pairs "
                  + " | ".join(f"{k} {v2[1]:.3f}" for k, v2 in pr.items())
                  + f" | tau {taus}", flush=True)

    print("\n=== variant selection (VALIDATION loss only) ===")
    sel = None
    for kind in variants:
        vs = np.array([r["val"] for r in results[kind]])
        print(f"{kind:10} val {vs.mean():.4f} ± {vs.std():.4f}")
        if sel is None or vs.mean() < sel[1]:
            sel = (kind, vs.mean())
    kind = sel[0]
    print(f"-> selected by validation: {kind}")

    # top up the selected variant to final-seeds
    while len(results[kind]) < args.final_seeds:
        si = len(results[kind])
        model, (vf, vl), v = train_seed(tr_b, va_b, inc_scale, args, kind, si)
        pr = eval_pairs(model, pair_trajs, depth_scale, dscale)
        taus = np.round(model.log_tau.exp().detach().cpu().numpy() * 24, 2)
        curve = model_decay_curve(model, tr_b, gap_pairs)
        results[kind].append({"val": v, "vf": vf, "vl": vl, "pairs": pr, "taus": taus, "curve": curve})
        print(f"  {kind} seed{si}: val {v:.4f} | pairs "
              + " | ".join(f"{k} {v2[1]:.3f}" for k, v2 in pr.items()), flush=True)

    print(f"\n=== final: {kind}, {len(results[kind])} seeds (matched pairs; distribution) ===")
    for pname in PAIRS:
        tr_ = results[kind][0]["pairs"][pname][0]
        pr = np.array([r["pairs"][pname][1] for r in results[kind]])
        print(f"{pname:10} TRUE {tr_:.3f} | {pr.mean():.3f} ± {pr.std():.3f}  "
              f"seeds {np.round(pr, 3)}")
    print(f"decay curve TRUE : {[round(v,3) for v in true_curve]}")
    for kind2 in [kind]:
        cc = np.array([r["curve"] for r in results[kind2]])
        print(f"decay curve MODEL: " + " ".join(f"{m2:.3f}±{s2:.3f}" for m2, s2 in zip(cc.mean(0), cc.std(0))))
    print("\n=== all variants (transparency; not used for selection) ===")
    for kind2 in variants:
        line = []
        for pname in PAIRS:
            pr = np.array([r["pairs"][pname][1] for r in results[kind2]])
            line.append(f"{pname} {pr.mean():.3f}±{pr.std():.3f}")
        taus = np.array([r["taus"] for r in results[kind2]])
        print(f"{kind2:10} " + " | ".join(line) + f" | tau {np.round(taus.mean(0),2)}")


if __name__ == "__main__":
    main()
