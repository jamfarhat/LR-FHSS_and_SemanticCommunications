# Organized Examples

This folder contains reproducible scripts for the main analyses discussed in the project.

## Scripts

- `network_size_sweep.py`
  - Sweeps number of devices and compares DR8, DR9 and Semantic.
  - Uses a quick profile by default (faster exploratory runs).
  - Full profile is available with `--full`.
  - Uses denser node-count sampling in both quick and full profiles.
  - Per-seed simulations run in parallel; configure with `--jobs`.
  - Plots: AoI, Mean Semantic Distortion, Energy Efficiency, Success Probability.
  - Includes trade-off plot: AoI vs Energy Efficiency (DR8/DR9/Semantic).
  - Saves raw and summarized CSV files.

- `single_user_distortion_trace.py`
  - Simulates one monitored process and compares temporal distortion behavior.
  - Uses the same base and semantic parameters as `network_size_sweep.py`.
  - Uses the same event times for DR8, DR9 and Semantic.
  - Default mode is periodic (uniform decision interval every `average_interval` seconds).
  - Optional mode: Poisson decision epochs via `INTERVAL_MODE = 'poisson'`.
  - Shows semantic distortion `D_k(t)`, adaptive threshold and TX/No-TX events.
  - Exports one CSV per protocol and one comparison figure.

## Run

From project root:

```bash
python examples/organized/network_size_sweep.py
python examples/organized/single_user_distortion_trace.py
```

Run full sweep:

```bash
python examples/organized/network_size_sweep.py --full
```

Control parallel workers:

```bash
python examples/organized/network_size_sweep.py --jobs 4
```

## Output

All outputs are saved with timestamped subfolders under `simulation_results/`:

- `simulation_results/network_sweep/<timestamp>/`
- `simulation_results/single_user_trace/<timestamp>/`

For `network_sweep`, the main figures are:

- `metrics_by_num_devices.png`
- `tradeoff_aoi_vs_energy.png`
