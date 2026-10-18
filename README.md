# Corrosion World Model · framework-only release

A compact source-code snapshot of the computational framework for investigating
history-conditioned localized iron dissolution and finite-memory latent dynamics.

**Status:** Research framework / partial code release, not a complete replication
package. This repository deliberately contains **no training logs, output tables,
experimental results, figures, datasets, trajectories, model checkpoints,
pretrained weights, or manuscript drafts**. It does not claim that the released
subset alone reproduces every number in the associated research manuscript.

## Included modules

| Path | Purpose |
|---|---|
| `wm/memory_dynamics.py` | Single-trajectory finite-memory chemical latent model |
| `wm/memory_dynamics_batched.py` | Batched training and multi-timescale latent dynamics |
| `wm/hist_vs_state.py` | State-only versus history-conditioned reference predictors |
| `tools/local_chem.py` | Geometry-aligned near-pit chemical features |
| `tools/faraday_depth.py` | Faraday conversion and boundary-profile resampling |
| `tools/gen_chained_hist_dataset.py` | Parameter sampling and chained simulation job preparation |
| `comsol/java/ChainedRunner2.java` | COMSOL Java runner for chained forcing schedules |
| `comsol/java/EnvMapCl3.java` | COMSOL Java parameter / environmental forcing setup |
| `tools/jrun.sh` | Example Linux invocation of a separately installed COMSOL API |
| `tests/smoke_test.py` | Dependency-light checks of the released model and units |

## Installation

Python 3.10+ and an environment with PyTorch are recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py
```

## COMSOL requirement

The Java utilities require a separately licensed installation of COMSOL
Multiphysics, its applicable modules, a valid model template, and access to
compatible COMSOL Java libraries. The original `.mph` model, XML export and
COMSOL executable are **not distributed**.

Example (Linux, after providing your own model and jobs file):

```bash
COMSOL_ROOT=/path/to/comsol64 bash tools/jrun.sh ChainedRunner2 path/to/template.mph path/to/jobs.csv
```

For the Python job generator, put your legally obtained template at
`comsol/templates/q235_env_v3.mph`. It generates job manifests under
`data/chained_hist/` locally; these folders are ignored by Git.

## Data expectations

The training scripts expect trajectory-export files and `data/chained_hist/params.csv`
in the formats defined by the original project. Those files are intentionally
absent. Some historical scripts also refer to demonstration matched-pair
identifiers and cannot be executed end-to-end without the original private data.

**Important:** This framework uses a fixed-film, reaction-gated,
potentiostatically driven localized iron-dissolution simulator. It must not be
misrepresented as a validated open-circuit atmospheric corrosion model.

## Publication and reproducibility

This code-only release is intended to disclose the main software structure.
Independent reproduction of the paper's quantitative results additionally requires
the specific numerical model, data splits, pair manifest, predictions, parameter
files, and experimental outputs. Repository publication alone does not satisfy
those data-availability obligations.

## License

No open-source license has been selected for this curated release. All rights
remain with the respective copyright holders pending an explicit licensing
decision. If redistribution or reuse is intended, please add a license after
confirming agreement with all contributors.
