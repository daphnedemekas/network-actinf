## network-actinf

Active inference on networks: simulations and experiments for Ising-like network models and a dual-agent Prisoners' Dilemma.

### Structure
- `ising_model/`: Ising-like network dynamics and sweeps
  - `simulation.py`: core `Simulation` and `SimulationVectorized`
  - `demo_vectorized.py`: vectorized demo; saves plots to `tmp_out/`
  - `demo_sim.py`: simple sweep/plot demo
  - `ERconn_po_sweep.py`: ER connectivity × po sweep with pickled results
  - `math_utils.py`: numerically-stable helpers
- `network/`: reusable network generators
  - `config.py`: `graph_generation_fns` map
  - `networks.py`: helpers to construct/draw multiple networks
- `prisoners_dilemma/`: dual-agent experiments (via `pymdp`)
  - `utils.py`: build agents and run dual simulations/sweeps
  - `functions.py`: low-level EFE helpers (for reference)
  - `sweep.py`: stochastic sweep over learning rates; writes under `stochastic/`
- `scripts/`: CLI entrypoints
  - `run_experiments.py`: unified experiments runner

### Installation
1) Python 3.10+
2) Install deps:
```bash
pip install -r requirements.txt
```

### Quickstart
List available experiments:
```bash
python scripts/run_experiments.py --help
```

Run an Ising ER sweep (writes results to `runs/er_sweep_default`):
```bash
python scripts/run_experiments.py ising-er-sweep --N 500 --T 1000 --num-trials 50 --run-name runs/er_sweep_default
```

Run the vectorized demo (plots to `tmp_out/`):
```bash
python scripts/run_experiments.py ising-demo-vectorized --T 1000
```

Run a PD stochastic sweep chunk (writes to `prisoners_dilemma/stochastic/`):
```bash
python scripts/run_experiments.py pd-stochastic-sweep --task-start 50 --task-end 55 --T 2000 --trials 100
```

### Experiments
- `ising-er-sweep`: sweeps ER `p` and likelihood `po`; pickles per-config results with averages of VFE components and branching parameter m.
- `ising-demo-vectorized`: vectorized simulation producing activity/spin heatmaps.
- `pd-stochastic-sweep`: runs batches of dual-agent learning-rate sweeps.

### Reproducibility
Set seeds via `--seed` on commands that expose it. Outputs are written under `runs/` or module-specific folders.

### Notes
- Some notebooks and figures under `prisoners_dilemma/dual_agent_results/` illustrate analysis but are not required to run experiments.
- Large sweeps can be slow; start with smaller `--num-trials`, `--T`, and narrower ranges.

### Plots from `ising_model/demo_sim.py`
Running the demo script will generate several figures under `ising_simulation_outputs/`.

Run the demo:
```bash
python ising_model/demo_sim.py
```

What each figure shows:
- **Network visualization (`network_visualization.png`)**: Final-time snapshot of the graph with nodes colored by spin state (Red = 1/down, Blue = 0/up). Uses a spring layout to reveal community/degree structure.
- **Phase space trajectories (`phase_trajectories.png`)**: Time traces of posterior beliefs (φ) for a handful of nodes. Lines are color-graded across time with markers for start (green) and end (red).
- **Spin–spin correlation matrix (`correlation_matrix.png`)**: Pearson correlation of node activities over a chosen window. Values range from -1 (anti-correlated) to +1 (correlated). When the network is large, a subset of nodes is visualized.
- **Avalanche analysis (`avalanche_analysis.png`)**: Four panels:
  - Avalanche size over time with a threshold line.
  - Distribution of non-zero avalanche sizes (log-scaled y-axis).
  - Activity heatmap around the largest avalanche with the peak marked.
  - Autocorrelation of avalanche sizes across lags.
- **Energy landscape (`energy_landscape.png`)**: Four panels capturing macroscopic dynamics:
  - Energy vs. time (interaction energy from current spins and adjacency).
  - Magnetization vs. time (mean spin).
  - Energy–magnetization phase plot, color-coded by time.
  - Energy distribution histogram.
- **Synchronization analysis (`synchronization_analysis.png`)**: Four panels characterizing phase coordination (using φ as phases):
  - Global order parameter |mean(exp(i·2π·φ))| over time.
  - Example phase time series for a small node subset.
  - Mean pairwise phase-difference matrix across a subsampled time grid.
  - Distribution of the order parameter.
- **Activity heatmap (`activity_heatmap.png`)**: Heatmap of spin states over time (x-axis = time, y-axis = nodes; subsampled rows for readability). Light/dark encodes spin state.
- **Regime comparison (`regime_comparison.png`)**: Stacked activity heatmaps for multiple likelihood values `p_\mathcal{O}` to contrast dynamical regimes at fixed network topology.

All figures are saved to `ising_simulation_outputs/` and displayed during the run.
