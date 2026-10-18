# Simulation data

This directory provides the simulation parameter manifest and a representative set of genuine COMSOL export snapshots for the finite-memory corrosion modeling workflow.

## Files

| File | Description |
| --- | --- |
| `chained_hist/params.csv` | **120 original simulation schedules** (`h0000`–`h0119`) containing temperature parameters, film thickness, wet-window durations, chloride schedules, frame times and output prefixes. The `out` column has been normalized to repository-relative paths. |
| `examples/h0000_field_snapshots.csv` | Field values at **four actual simulation times** (0, 0.881, 2.000 and 3.954 days) for 318 spatial nodes in one original COMSOL trajectory. |
| `examples/h0000_surface_current.csv` | Interfacial current density at the same four times for 39 electrode-boundary nodes from the same trajectory. |

The COMSOL data are obtained from the original `h0000` exports without interpolation or model-generated replacements. Field snapshots are selected temporal slices of one complete trajectory, **not the full 120-trajectory training tensor**.

## Column definitions

- `T`: mean ambient temperature in °C; `dT`: temperature-amplitude parameter in °C.
- `film`: prescribed electrolyte film thickness in µm.
- `wets`: comma-separated active intervals for each of four one-day cycles (days).
- `cl`: comma-separated chloride-loading inputs (mg/m²) in the archived job generator.
- `tlist`: comma-separated COMSOL export times (days).
- `out`: repository-relative simulation run prefix; the Python training scripts construct the actual run location using `data/chained_hist/runs/<name>`.
- `r_m`, `z_m`: 2D axisymmetric spatial coordinates (m).
- `cFe_mol_m3`, `cCl_mol_m3`: local ion concentrations (mol/m³).
- `pH`: local COMSOL calculated pH (dimensionless).
- `electrolyte_potential_V`: electrolyte potential (V).
- `iloc_A_m2`: electrode interfacial dissolution current density (A/m²).

## Running the training pipeline

To train over the full set of 120 schedules, populate `data/chained_hist/runs/` with corresponding original exports from a licensed COMSOL model. Each run uses the prefix `hNNNN` and the file suffixes `_iloc.txt`, `_fe2plus.txt`, `_ph.txt`, `_chloride.txt`, and `_potential.txt`. The provided snapshots are useful for inspecting the data schema and physical scales, but are **not a complete training corpus**.

**Model boundary:** Fixed electrolyte film, reaction-gated intervals and an applied-electrode-potential anodic dissolution setup; this is not a naturally open-circuit atmospheric-corrosion model.
