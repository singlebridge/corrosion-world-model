#!/usr/bin/env python3
"""Step 5 (simulation side): is history needed beyond the current observable state?

Train two predictive models on the chained-history dataset (data/chained_hist):

  state-only   : current depth profile + current/future environment + dt -> future increment
  history-aware: same + the last K (depth profile, env) frames

Then evaluate on the MATCHED PAIRS (data/histed/{pS_A, pS_B0875}): at the switch
t=2.0 the two chains have (nearly) identical current depth but different pasts and
DIFFERENT true futures (increment ratio 1.81x).  A state-only model cannot
separate them; a history-aware model can, if the observable history carries the
hidden-state information.

Usage: .venv/bin/python wm/hist_vs_state.py [--epochs 3000] [--K 6]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from faraday_depth import parse_iloc, faraday, resample_arc  # noqa: E402
from local_chem import extract_features as extract_chem  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data loading
def load_traj(prefix):
    t, coords, iloc = parse_iloc(str(prefix) + "_iloc.txt")
    depth, cr = faraday(iloc, t)
    _, dp = resample_arc(coords, depth)
    _, crp = resample_arc(coords, cr)
    return t, dp, crp  # times[d], depth[m, nbins, F], cr[mm/a, nbins, F]


def env_features(t, T, dT, cl_period, film, dt):
    """Environment feature vector for horizon dt starting at t (days)."""
    return np.array([
        T / 25.0, dT / 6.0, float(np.cos(2 * np.pi * t)), float(np.sin(2 * np.pi * t)),
        np.log1p(cl_period) / 7.0, film / 10.0, dt,
    ], dtype=np.float32)


def memory_features(t, wet, cl_period, dep):
    """Explicit, OBSERVABLE history summary (no hidden fields).

    Per frame: time since the previous wet end; time since the current/most
    recent wet start; recent max-depth growth over 0.1/0.3/0.6 d; chloride load
    of the previous wet period (the 'hot/cold' history signal).
    """
    F = len(t)
    dmax = dep.max(axis=0)
    feats = np.zeros((F, 6), np.float32)
    prev_end = None; cur_start = None; prev_cl = 0.0
    wet_prev = False
    for j in range(F):
        tj = t[j]
        if wet[j] and not wet_prev:
            cur_start = tj
        if (not wet[j]) and wet_prev:
            prev_end = t[j - 1]
            prev_cl = float(cl_period[j - 1])
        wet_prev = wet[j]
        feats[j, 0] = 30.0 if prev_end is None else (tj - prev_end)
        feats[j, 1] = 30.0 if cur_start is None else (tj - cur_start)
        for li, lag in enumerate((0.1, 0.3, 0.6)):
            k = None
            for kk in range(j, -1, -1):
                if t[kk] <= tj - lag + 1e-9:
                    k = kk; break
            feats[j, 2 + li] = 0.0 if k is None else float(dmax[j] - dmax[k])
        feats[j, 5] = prev_cl
    return feats


def build_samples(trajs, K, horizon_choices=(1, 2, 3, 5, 8, 12)):
    """trajs: list of dicts with t, depth(um), cl_period(F), T,dT,film, wet_mask(F)."""
    X_cur, X_hist, X_eng, X_ora, X_chem, Y, meta = [], [], [], [], [], [], []
    for ti, tr in enumerate(trajs):
        t = tr["t"]; dep = tr["depth"]  # (128, F) um
        F = len(t)
        wet = tr["wet"]
        mf = memory_features(t, wet, tr["cl_period"], dep)
        for j in range(F):
            if not wet[j]:
                continue
            for h in horizon_choices:
                k = j + h
                if k >= F or not wet[k]:
                    continue
                dt = float(t[k] - t[j])
                if dt <= 0:
                    continue
                cur = np.concatenate([
                    dep[:, j],
                    env_features(t[j], tr["T"], tr["dT"], tr["cl_period"][j], tr["film"], dt),
                ])
                # history: last K frames (pad with current)
                hs = []
                for z in range(1, K + 1):
                    jj = max(0, j - z)
                    hs.append(np.concatenate([
                        dep[:, jj],
                        env_features(t[jj], tr["T"], tr["dT"], tr["cl_period"][jj], tr["film"],
                                     float(t[j] - t[jj])),
                    ]))
                hist = np.concatenate([cur] + hs)
                X_cur.append(cur); X_hist.append(hist)
                X_eng.append(np.concatenate([cur, mf[j]]))
                X_ora.append(np.concatenate([cur, tr["hidden"][j]]))
                X_chem.append(np.concatenate([cur, mf[j], tr["local"][j]]))
                Y.append(dep[:, k] - dep[:, j])
                meta.append((ti, j, k, dt))
    return (np.asarray(X_cur, np.float32), np.asarray(X_hist, np.float32),
            np.asarray(X_eng, np.float32), np.asarray(X_ora, np.float32),
            np.asarray(X_chem, np.float32), np.asarray(Y, np.float32), meta)


def hidden_stats(prefix):
    """Oracle features: per-frame stats of the hidden fields (Fe2+, pH, potential).

    Returns (F, 12): [fe2 max/mean/std/p90, ph max/mean/std/p90, pot max/mean/std/p90].
    """
    out = []
    for ch in ("fe2plus", "ph", "potential"):
        header = None; rows = []
        for ln in open(str(prefix) + f"_{ch}.txt"):
            if ln.startswith("%"):
                if "@ t=" in ln: header = ln
                continue
            if ln.strip(): rows.append([float(x) for x in ln.split()])
        v = np.array(rows)[:, 2:]  # (nodes, F)
        out.append(np.stack([
            v.max(0), v.mean(0), v.std(0), np.percentile(v, 90, axis=0),
        ], axis=1))  # (F, 4)
    return np.concatenate(out, axis=1).astype(np.float32)


def read_params(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["name"]] = r
    return rows


def assemble(params, runs_dir, require_complete=True):
    out = []
    for name, r in params.items():
        prefix = Path(runs_dir) / name
        if not Path(str(prefix) + "_iloc.txt").exists():
            continue
        t, dp, cr = load_traj(prefix)
        wets = [float(x) for x in r["wets"].split(",")]
        cls = [float(x) for x in r["cl"].split(",")]
        cycle = 1.0
        cl_period = np.zeros(len(t)); wet = np.zeros(len(t), bool)
        for i, w in enumerate(wets):
            m = (t >= i * cycle - 1e-9) & (t <= i * cycle + w + 1e-9)
            wet |= m
            cl_period[m] = cls[i]
        cache = Path(str(runs_dir)) / "_chem_cache.npz"
        ck = f"{name}_local"
        if cache.exists():
            cz = np.load(cache)
            if ck in cz and f"{name}_glob" in cz:
                loc, glob = cz[ck], cz[f"{name}_glob"]
            else:
                loc, glob = extract_chem(prefix)
        else:
            loc, glob = extract_chem(prefix)
        out.append({"name": name, "t": t, "depth": dp * 1e6, "cr": cr * 1e3,
                    "wet": wet, "cl_period": cl_period,
                    "local": loc, "hidden": glob,
                    "T": float(r["T"]), "dT": float(r["dT"]), "film": float(r["film"])})
    # persist the (slow) chemistry features for fast iteration
    cache = Path(str(runs_dir)) / "_chem_cache.npz"
    store = {}
    for tr_ in out:
        store[f"{tr_['name']}_local"] = tr_["local"]
        store[f"{tr_['name']}_glob"] = tr_["hidden"]
    np.savez_compressed(cache, **store)
    return out


# ---------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, din, dout=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dout))

    def forward(self, x):
        return self.net(x)


class DualHead(nn.Module):
    """Student world model: observable -> inferred hidden chemistry -> future."""
    def __init__(self, din, dlat=12, dout=128, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(din, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.lat = nn.Linear(hidden, dlat)
        self.head = nn.Sequential(
            nn.Linear(hidden + dlat, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dout))

    def forward(self, x):
        h = self.trunk(x)
        z = self.lat(h)
        return z, self.head(torch.cat([h, z], -1))


def train_dual(Xtr, Ztr, Ytr, Xva, Zva, Yva, epochs, lam=1.0, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = DualHead(Xtr.shape[1], dlat=Ztr.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, device=DEV); Zt = torch.tensor(Ztr, device=DEV); Yt = torch.tensor(Ytr, device=DEV)
    Xv = torch.tensor(Xva, device=DEV); Zv = torch.tensor(Zva, device=DEV); Yv = torch.tensor(Yva, device=DEV)
    n = len(Xt); best = (1e9, None)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            zp, yp = m(Xt[idx])
            loss = ((yp - Yt[idx]) ** 2).mean() + lam * ((zp - Zt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0 or ep == epochs - 1:
            with torch.no_grad():
                zp, yp = m(Xv)
                v = ((yp - Yv) ** 2).mean().item() + lam * ((zp - Zv) ** 2).mean().item()
            if v < best[0]:
                best = (v, {k: vv.clone() for k, vv in m.state_dict().items()})
    m.load_state_dict(best[1])
    with torch.no_grad():
        zp, yp = m(Xv)
        vf = ((yp - Yv) ** 2).mean().item()
        vz = ((zp - Zv) ** 2).mean().item()
    return m, vf, vz


class GRUPred(nn.Module):
    """History encoder: GRU over the frame sequence (oldest -> current)."""
    def __init__(self, dframe=135, hidden=128, dout=128):
        super().__init__()
        self.enc = nn.Linear(dframe, 96)
        self.gru = nn.GRU(96, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + dframe, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, dout))

    def forward(self, xseq):
        h, _ = self.gru(self.enc(xseq))
        return self.head(torch.cat([h[:, -1], xseq[:, -1]], -1))


def train_model(Xtr, Ytr, Xva, Yva, epochs, seed=0, lr=1e-3, bs=256, model_fn=None):
    torch.manual_seed(seed)
    m = (model_fn(Xtr) if model_fn else MLP(Xtr.shape[1])).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr, device=DEV); Ytr_t = torch.tensor(Ytr, device=DEV)
    Xva_t = torch.tensor(Xva, device=DEV); Yva_t = torch.tensor(Yva, device=DEV)
    n = len(Xtr_t)
    best = (1e9, None)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = m(Xtr_t[idx])
            loss = ((pred - Ytr_t[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0 or ep == epochs - 1:
            with torch.no_grad():
                v = ((m(Xva_t) - Yva_t) ** 2).mean().item()
            if v < best[0]:
                best = (v, {k: vv.clone() for k, vv in m.state_dict().items()})
    m.load_state_dict(best[1])
    return m, best[0]


# ---------------------------------------------------------------- matched pair
def pair_future(prefix, switch=2.0, end=2.9):
    t, dp, cr = load_traj(prefix)
    i = int(np.argmin(np.abs(t - switch)))
    j = int(np.argmin(np.abs(t - end)))
    return t, dp * 1e6, i, j


def pair_features(t, dep, j, K, T, dT, cl_period_fn, film, dt):
    """Build (current, history) features at frame j for the pair."""
    def ef(tt, cp, dtv):
        return env_features(tt, T, dT, cp, film, dtv)
    cur = np.concatenate([dep[:, j], ef(t[j], cl_period_fn(t[j]), dt)])
    hs = []
    for z in range(1, K + 1):
        jj = max(0, j - z)
        hs.append(np.concatenate([dep[:, jj], ef(t[jj], cl_period_fn(t[jj]), float(t[j] - t[jj]))]))
    return np.concatenate([cur] + hs)[None, :], cur[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT = ROOT / "data/chained_hist"
    params = read_params(OUT / "params.csv")
    trajs = assemble(params, OUT / "runs")
    print(f"loaded {len(trajs)} complete trajectories")
    if args.limit:
        trajs = trajs[:args.limit]

    # trajectory-level split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(trajs))
    n_val = max(2, int(0.15 * len(trajs)))
    val_ids = set(idx[:n_val].tolist())
    tr = [t for i, t in enumerate(trajs) if i not in val_ids]
    va = [t for i, t in enumerate(trajs) if i in val_ids]
    print(f"train {len(tr)} / val {len(va)} trajectories")

    Xc_tr, Xh_tr, Xe_tr, Xo_tr, Xm_tr, Y_tr, _ = build_samples(tr, args.K)
    Xc_va, Xh_va, Xe_va, Xo_va, Xm_va, Y_va, _ = build_samples(va, args.K)
    dscale = Y_tr.std() + 1e-9
    Y_tr /= dscale; Y_va /= dscale
    # global feature normalization (shared)
    mu = Xc_tr.mean(0); sd = Xc_tr.std(0) + 1e-6

    def norm(x):
        # x shape (..., d): first d_cur columns are the 'current' block; history
        # blocks repeat with their own scale ~ same distribution -> normalize by
        # tiling mu/sd over blocks.
        d = x.shape[-1]; dcur = len(mu)
        reps = d // dcur
        m = np.tile(mu, reps)[None, :]; s = np.tile(sd, reps)[None, :]
        return (x - m) / s

    print(f"samples: train {len(Xc_tr)}, val {len(Xc_va)}")
    Xc_tr_n, Xc_va_n = norm(Xc_tr), norm(Xc_va)
    Xh_tr_n, Xh_va_n = norm(Xh_tr), norm(Xh_va)
    # engineered-history normalizer (own statistics)
    mu_e = Xe_tr.mean(0); sd_e = Xe_tr.std(0) + 1e-6
    def norm_e(x):
        return (x - mu_e) / sd_e
    m_state, v_state = train_model(Xc_tr_n, Y_tr, Xc_va_n, Y_va, args.epochs)
    mu_o = Xo_tr.mean(0); sd_o = Xo_tr.std(0) + 1e-6
    def norm_o(x):
        return (x - mu_o) / sd_o
    m_eng, v_eng = train_model(norm_e(Xe_tr), Y_tr, norm_e(Xe_va), Y_va, args.epochs)
    m_ora, v_ora = train_model(norm_o(Xo_tr), Y_tr, norm_o(Xo_va), Y_va, args.epochs)
    mu_m = Xm_tr.mean(0); sd_m = Xm_tr.std(0) + 1e-6
    def norm_m(x):
        return (x - mu_m) / sd_m
    m_chem, v_chem = train_model(norm_m(Xm_tr), Y_tr, norm_m(Xm_va), Y_va, args.epochs)
    # ---- student: infer the hidden local chemistry from OBSERVABLE inputs only ----
    nloc = Xm_tr.shape[1] - Xc_tr.shape[1] - 6      # local chemistry block size
    dobs = Xm_tr.shape[1] - nloc
    def norm_obs(x):
        return (x - mu_m[:dobs]) / sd_m[:dobs]
    def norm_lat(x):
        return (x - mu_m[dobs:]) / sd_m[dobs:]
    m_wm, v_wm, v_wmz = train_dual(norm_obs(Xm_tr[:, :dobs]), norm_lat(Xm_tr[:, dobs:]), Y_tr,
                                   norm_obs(Xm_va[:, :dobs]), norm_lat(Xm_va[:, dobs:]), Y_va,
                                   args.epochs)
    m_hist, v_hist = train_model(Xh_tr_n, Y_tr, Xh_va_n, Y_va, args.epochs)
    # GRU consumes frames oldest -> current: (N, K+1, d_frame)
    dframe = Xc_tr_n.shape[1]
    seq_tr = Xh_tr_n.reshape(len(Xh_tr_n), -1, dframe)[:, ::-1, :].copy()
    seq_va = Xh_va_n.reshape(len(Xh_va_n), -1, dframe)[:, ::-1, :].copy()
    m_gru, v_gru = train_model(seq_tr, Y_tr, seq_va, Y_va, args.epochs,
                               model_fn=lambda x: GRUPred(x.shape[-1]))

    # capacity-controlled linear baselines
    from sklearn.linear_model import Ridge
    ridge_state = Ridge(alpha=10.0).fit(Xc_tr_n, Y_tr)
    ridge_hist = Ridge(alpha=10.0).fit(Xh_tr_n, Y_tr)
    ridge_eng = Ridge(alpha=10.0).fit(norm_e(Xe_tr), Y_tr)
    ridge_ora = Ridge(alpha=10.0).fit(norm_o(Xo_tr), Y_tr)
    ridge_chem = Ridge(alpha=10.0).fit(norm_m(Xm_tr), Y_tr)
    v_rs = float(((ridge_state.predict(Xc_va_n) - Y_va) ** 2).mean())
    v_rh = float(((ridge_hist.predict(Xh_va_n) - Y_va) ** 2).mean())
    v_re = float(((ridge_eng.predict(norm_e(Xe_va)) - Y_va) ** 2).mean())
    v_ro = float(((ridge_ora.predict(norm_o(Xo_va)) - Y_va) ** 2).mean())
    v_rc = float(((ridge_chem.predict(norm_m(Xm_va)) - Y_va) ** 2).mean())

    print(f"\nvalidation MSE (profile, normalized):")
    print(f"  {'':14}{'MLP':>10}{'ridge':>10}{'GRU':>10}")
    print(f"  state-only    {v_state:10.5f}{v_rs:10.5f}{'-':>10}")
    print(f"  eng-history   {v_eng:10.5f}{v_re:10.5f}{'-':>10}")
    print(f"  full-history  {v_hist:10.5f}{v_rh:10.5f}{v_gru:10.5f}")
    print(f"  ORACLE(hidden){v_ora:10.5f}{v_ro:10.5f}{'-':>10}")
    print(f"  ORACLE(local) {v_chem:10.5f}{v_rc:10.5f}{'-':>10}")
    print(f"  LATENT-WM     {v_wm:10.5f}{'-':>10}{'-':>10}   (inferred-latent MSE {v_wmz:.4f})")

    # matched pairs at three dry-gap lengths (true A/B future-increment ratios)
    PAIRS = [
        ("gap2.4h", 2.9, [
            ("A", ROOT / "data/histed/pS_A", [0.9, 0.9, 0.9], [100, 800, 400]),
            ("B", ROOT / "data/histed/pS_B0875", [0.875, 0.9, 0.9], [800, 100, 400]),
        ]),
        ("gap4.8h", 2.9, [
            ("A", ROOT / "data/histed/sweep2/A_g020", [0.9, 0.8, 0.9], [100, 800, 400]),
            ("B", ROOT / "data/histed/sweep2/Bm_g020", [0.8331, 0.8, 0.9], [800, 100, 400]),
        ]),
        ("gap16.8h", 2.7, [
            ("A", ROOT / "data/histed/pL_A", [0.3, 0.3, 0.7], [100, 800, 400]),
            ("B", ROOT / "data/histed/pL_B", [0.3, 0.3, 0.7], [800, 100, 400]),
        ]),
    ]
    print("\n=== matched pairs (switch t=2.0) ===")
    print(f"{'pair':10} {'tag':>2} {'true inc':>9} {'MLPs':>9} {'MLPe':>9} {'MLPh':>9}"
          f" {'RGs':>9} {'RGe':>9} {'RGh':>9} {'GRU':>9} {'ORACLE':>9} {'ORA-RG':>9} {'CHEM':>9} {'CHEM-RG':>9} {'WM':>9} {'zErr':>7}")
    summary = {}
    for pname, end, members in PAIRS:
        res = {}
        for tag, prefix, wets, cls in members:
            t, dep, i, j = pair_future(prefix, 2.0, end)
            def cl_fn(tt, cls=cls, wets=wets):
                # match the training convention: dry frames carry zero load
                for k, w in enumerate(wets):
                    if k * 1.0 - 1e-9 <= tt <= k * 1.0 + w + 1e-9:
                        return cls[k]
                return 0.0
            # explicit observable-history features from the pair's own frame grid
            wet_p = np.zeros(len(t), bool); clp = np.zeros(len(t))
            for k, w in enumerate(wets):
                m = (t >= k * 1.0 - 1e-9) & (t <= k * 1.0 + w + 1e-9)
                wet_p |= m; clp[m] = cls[k]
            mf = memory_features(t, wet_p, clp, dep)
            dt = float(t[j] - t[i])
            xh, xc = pair_features(t, dep, i, args.K, 25.0, 6.0, cl_fn, 10.0, dt)
            xe = np.concatenate([xc[0], mf[i]])[None, :]
            hid = hidden_stats(prefix)
            xo = np.concatenate([xc[0], hid[i]])[None, :]
            loc_chem, _ = extract_chem(prefix)
            xm = np.concatenate([xc[0], mf[i], loc_chem[i]])[None, :]
            xobs = xm[:, :dobs]
            ztrue = xm[:, dobs:]
            with torch.no_grad():
                ps = m_state(torch.tensor(norm(xc).astype(np.float32), device=DEV)).cpu().numpy()[0] * dscale
                pe = m_eng(torch.tensor(norm_e(xe).astype(np.float32), device=DEV)).cpu().numpy()[0] * dscale
                ph = m_hist(torch.tensor(norm(xh).astype(np.float32), device=DEV)).cpu().numpy()[0] * dscale
                seq = torch.tensor(norm(xh).reshape(1, -1, dframe)[:, ::-1, :].astype(np.float32).copy(), device=DEV)
                pg = m_gru(seq).cpu().numpy()[0] * dscale
                po = m_ora(torch.tensor(norm_o(xo).astype(np.float32), device=DEV)).cpu().numpy()[0] * dscale
                pm = m_chem(torch.tensor(norm_m(xm).astype(np.float32), device=DEV)).cpu().numpy()[0] * dscale
                zhat, pw = m_wm(torch.tensor(norm_obs(xobs).astype(np.float32), device=DEV))
                pw = pw.cpu().numpy()[0] * dscale
                zerr = float((zhat - torch.tensor(norm_lat(ztrue), device=DEV)).abs().mean().item())
            prs = ridge_state.predict(norm(xc))[0] * dscale
            pre = ridge_eng.predict(norm_e(xe))[0] * dscale
            prh = ridge_hist.predict(norm(xh))[0] * dscale
            pro = ridge_ora.predict(norm_o(xo))[0] * dscale
            prm = ridge_chem.predict(norm_m(xm))[0] * dscale
            true_inc = dep[:, j] - dep[:, i]
            preds = {"MLPs": ps, "MLPe": pe, "MLPh": ph,
                     "RGs": prs, "RGe": pre, "RGh": prh, "GRU": pg,
                     "ORACLE": po, "ORA-RG": pro, "CHEM": pm, "CHEM-RG": prm,
                     "WM": pw}
            res[tag] = {"preds": preds, "true": true_inc}
            print(f"{pname:10} {tag:>2} {true_inc.max():9.4f} {ps.max():9.4f} {pe.max():9.4f} {ph.max():9.4f}"
                  f" {prs.max():9.4f} {pre.max():9.4f} {prh.max():9.4f} {pg.max():9.4f}"
                  f" {po.max():9.4f} {pro.max():9.4f} {pm.max():9.4f} {prm.max():9.4f}"
                  f" {pw.max():9.4f} {zerr:7.3f}")
        tr = float(res["A"]["true"].max() / res["B"]["true"].max())
        ratio = {name: float(res["A"]["preds"][name].max() / res["B"]["preds"][name].max())
                 for name in res["A"]["preds"]}
        summary[pname] = (tr, ratio)
        print("  -> A/B ratio: TRUE %.3f | " % tr
              + " | ".join(f"{k} {v:.3f}" for k, v in ratio.items()))
    print("\n=== summary: A/B future-increment ratio (target = TRUE) ===")
    for pname, (tr, ratio) in summary.items():
        parts = " | ".join(f"{k} {v:.3f} ({abs(v-tr)/tr:5.1%})" for k, v in ratio.items())
        print(f"{pname:10} TRUE {tr:.3f} | {parts}")


if __name__ == "__main__":
    main()
