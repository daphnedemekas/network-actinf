# Ising Model Demo: Figures and Descriptions


## How to Run
```bash
python ising_model/demo_sim.py
```
All images will be written to `ising_simulation_outputs/`.

## Mathematical Background

The simulation implements Ising-like dynamics with variational free energy calculations:
- **Spin states**: Binary variables (0/up, 1/down) representing node states
- **Posterior beliefs (φ)**: Probability that a node is in the down state
- **Observation likelihood (po)**: Controls system sensitivity and regime transitions
- **Network effects**: Neighboring spins influence each other through the adjacency matrix

## Visualizations

### Network Visualization
Final-time snapshot of the graph with nodes colored by spin state (Red = 1/down, Blue = 0/up). Shows the spatial structure of the network and how spin states cluster or distribute across the topology at equilibrium.

![Network visualization](ising_simulation_outputs/network_visualization.png)

### Phase Space Trajectories
Time traces of posterior beliefs (φ) for a small set of nodes. Color gradient progresses through time; start points in green, end points in red. Reveals how individual nodes evolve their beliefs over time and whether they converge to stable states or exhibit complex dynamics.

![Phase space trajectories](ising_simulation_outputs/phase_trajectories.png)

### Spin-Spin Correlation Matrix
Pearson correlation of node activities over a selected time window. Values range from -1 (anti-correlated) to +1 (perfectly correlated). Large networks are subsampled for readability. Identifies which nodes tend to synchronize their behavior and reveals community structure or clustering patterns.

![Correlation matrix](ising_simulation_outputs/correlation_matrix.png)

### Avalanche Analysis
Four panels analyzing cascade dynamics: (1) avalanche size over time with detection threshold; (2) distribution of non-zero avalanche sizes (log-scaled y-axis) - power-law distributions indicate criticality; (3) activity heatmap around the largest avalanche with peak marked in red; (4) autocorrelation of avalanche sizes across time lags showing temporal dependencies.

![Avalanche analysis](ising_simulation_outputs/avalanche_analysis.png)

### Energy Landscape
Four panels capturing macroscopic system dynamics: (1) interaction energy vs time (computed from spin configurations and network structure); (2) magnetization (average spin state) vs time; (3) energy-magnetization phase plot colored by time progression; (4) energy distribution showing preferred system states.

![Energy landscape](ising_simulation_outputs/energy_landscape.png)

### Synchronization Analysis
Four panels characterizing phase coordination using φ as phases: (1) global order parameter |⟨exp(i·2π·φ)⟩| measuring collective synchronization; (2) sample node phase time series showing individual dynamics; (3) mean pairwise phase-difference matrix revealing synchronization clusters; (4) distribution of the order parameter across time.

![Synchronization analysis](ising_simulation_outputs/synchronization_analysis.png)

### Activity Heatmap
Heatmap of spin states over time (x-axis = time, y-axis = nodes; subsampled for readability). Grayscale intensity encodes spin state (light = up, dark = down). Shows temporal patterns, propagation of activity, and identifies periods of high/low system activity.

![Activity heatmap](ising_simulation_outputs/activity_heatmap.png)

### Regime Comparison
Stacked activity heatmaps across different observation likelihood values (po) to contrast dynamical regimes on identical network topology. Demonstrates how the observation parameter controls system behavior: low po → random/disordered, high po → ordered/synchronized states.

![Regime comparison](ising_simulation_outputs/regime_comparison.png)

## Key Insights

### Behavioral Regimes
- **Low po values** (~0.5): Random, disordered dynamics with weak correlations
- **Medium po values** (~0.6): Transitional regime with emerging structure
- **High po values** (~0.75): Ordered, synchronized behavior with strong correlations

### Critical Phenomena
- **Avalanche distributions**: Power-law scaling suggests the system operates near criticality
- **Phase transitions**: Distinct changes in correlation structure and synchronization as po varies
- **Energy landscapes**: Multiple attractors and metastable states emerge in different regimes

### Network Effects
- **Topology matters**: Network structure influences correlation patterns and synchronization
- **Local vs global**: Individual node dynamics couple to produce collective behaviors
- **Emergent properties**: System-level patterns not present in isolated nodes

## Applications

This analysis framework is useful for:
- **Neuroscience**: Modeling neural network dynamics and criticality
- **Social systems**: Understanding opinion dynamics and consensus formation
- **Complex networks**: Studying phase transitions in networked systems
- **Active inference**: Analyzing belief propagation and collective decision-making
