"""Basic integrity checks for the source-only distribution.

Run: python tests/smoke_test.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wm"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
from memory_dynamics import MemDyn, DSLOW, DENV, DMEM
from memory_dynamics_batched import MemDyn as BatchedMemDyn
from faraday_depth import faraday, V_PER_AMP, resample_arc


def main():
    assert DSLOW == 128 and DENV == 6 and DMEM == 12
    F = 5
    with torch.no_grad():
        model = MemDyn(hidden=32).eval()
        slow = torch.zeros(F, DSLOW)
        env = torch.zeros(F, DENV)
        wet = torch.tensor([1, 1, 0, 0, 1], dtype=torch.float32)
        dt = torch.full((F-1,), 0.1)
        chem, mem = model(slow, env, wet, dt)
        assert chem.shape == (F, DMEM) and mem.shape == (F, DMEM)
        assert torch.isfinite(chem).all() and torch.isfinite(mem).all()
        assert model.predict(slow, env, chem, torch.full((F,),0.1)).shape == (F,DSLOW)
        batch = BatchedMemDyn(hidden=32)
        assert batch.log_tau.shape[0] == DMEM

    times = np.array([0.0, 1 / 86400])
    current = np.ones((2, 2), dtype=float)
    depth, rate = faraday(current, times)
    assert np.allclose(depth[:, 1], V_PER_AMP)
    assert np.all(rate > 0)
    coords = np.array([[0.,0.],[1.,0.]])
    _, resamp = resample_arc(coords, depth)
    assert resamp.shape == (128,2)
    print("smoke test: PASS (model tensors and Faraday conversion)")

if __name__ == "__main__":
    main()
