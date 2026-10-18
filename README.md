# Corrosion World Model

**Finite-memory latent dynamics for history-dependent localized corrosion**

This repository accompanies research on the role of local electrochemical states in history-dependent material degradation. It combines controlled multiphysics simulation, near-interface chemical-state extraction, history-conditioned prediction, and a structured latent dynamics model with learnable memory relaxation.

The central question is whether the currently observable corrosion geometry is sufficient to predict subsequent dissolution under a prescribed environment. The modeling approach augments the observable state with a finite-lived latent chemical memory that evolves with the exposure sequence.

## Research framework

### 1. Reaction–transport simulation

A COMSOL-based axisymmetric iron-dissolution model represents ionic transport, local electrochemical kinetics, Fe(II)-hydroxide reaction, and Faraday-based geometry evolution. Time-dependent exposure windows define controlled active and reaction-gated intervals.

### 2. History-dependent state analysis

Chained forcing schedules generate trajectories with different environmental histories. Observable geometry and extracted near-interface chemical features support comparisons between state-only and history-conditioned predictors.

### 3. Finite-memory latent dynamics

The predictive state is decomposed into an equilibrium-conditioned component and a dynamical memory component:

$$
 z_t^{\mathrm{chem}} = z_{\mathrm{eq}}(S_t,E_t) + m_t.
$$

During reaction-gated intervals, the memory is relaxed through channel-wise decay:

$$
 m_{t+\Delta t} = m_t \odot \exp(-\Delta t/\boldsymbol\tau).
$$

A learned transition handles active intervals, while a prediction head maps the current observable state, environment, and latent chemistry to subsequent damage evolution.

## Repository structure

```text
corrosion-world-model/
├── comsol/java/                 # COMSOL Java simulation runners
│   ├── ChainedRunner2.java
│   └── EnvMapCl3.java
├── wm/                          # Finite-memory and history baselines
│   ├── memory_dynamics.py
│   ├── memory_dynamics_batched.py
│   └── hist_vs_state.py
├── tools/                       # Simulation and chemical-feature utilities
│   ├── gen_chained_hist_dataset.py
│   ├── local_chem.py
│   ├── faraday_depth.py
│   └── jrun.sh
├── data/
│   ├── README.md                # Dataset description and physical units
│   ├── chained_hist/params.csv  # 120 chained-exposure simulation schedules
│   └── examples/                # Selected original COMSOL trajectory values
├── tests/smoke_test.py
├── requirements.txt
└── README.md
```

## Data

The repository provides **120 simulation-input schedules** (`data/chained_hist/params.csv`) and representative spatially resolved values from a COMSOL trajectory (`h0000`):

- [`data/examples/h0000_field_snapshots.csv`](data/examples/h0000_field_snapshots.csv): Fe²⁺, chloride, pH, and electrolyte potential at 318 spatial nodes for four exported times.
- [`data/examples/h0000_surface_current.csv`](data/examples/h0000_surface_current.csv): local interfacial current density at 39 boundary nodes for the same times.

These examples are derived from original simulation exports. Dataset columns, units, provenance, and the relationship between the parameter manifest and the trajectory-based training pipeline are described in [`data/README.md`](data/README.md).

## Getting started

Requirements: **Python 3.10+**, NumPy, SciPy, pandas, scikit-learn, and PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/smoke_test.py
```

To prepare additional chained-exposure simulations:

```bash
python tools/gen_chained_hist_dataset.py --n 120 --chunks 4
```

This command generates local simulation job tables. Executing those jobs requires a separately licensed installation of COMSOL Multiphysics with the relevant modules and a compatible `.mph` simulation model.

For numerical training, populate `data/chained_hist/runs/` with the corresponding multi-time COMSOL trajectory exports. The training modules read the schedule table and the exported field/current data from these locations.

## Modeling scope

The computational experiment represents **potentiostatically driven local anodic dissolution under a fixed electrolyte film**, with controlled intervals that gate the interfacial reaction. Electrolyte transport continues during the gated intervals. The configuration is used to study state sufficiency and chemical-memory dynamics; it is not a complete open-circuit atmospheric corrosion model.

## Software and citation

Research software and related data were developed for the corrosion world-model study. A versioned source archive and publication citation may be added when the associated manuscript is finalized.

COMSOL Multiphysics is proprietary software and must be obtained separately from its vendor. The repository does not grant a license to third-party software. Source-code reuse and redistribution remain subject to authorization from the respective copyright holders until a software license is confirmed.
