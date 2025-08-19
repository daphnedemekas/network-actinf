# Exponential Term Analysis: Figures and Descriptions

## How to Run
```bash
python ising_model/exponential_term_test.py
```
All images and arrays will be written to `exponential_analysis_output/`.

## Mathematical Function

The analysis explores the exponential term function:
```
f(ω, k) = 2 × (ω + ωc × exp(k × (1 - 2ω))) / (1 + exp(k × (1 - 2ω)))
```
where ω ∈ [0,1] is a belief parameter and k ≥ 0 controls precision/sharpness.

## Visualizations

### Omega Dependence
Exponential term as a function of omega (ω) for multiple precision values k. Shows how the response changes across the probability simplex. Each colored line represents a different k value, revealing the transition from linear behavior (low k) to nonlinear, sigmoid-like responses (high k).

![Omega dependence](exponential_analysis_output/omega_dependence.png)

### Precision (k) Dependence
Exponential term as a function of precision k for several fixed omega (ω) values. Reveals how sharpening the precision affects the mapping. Different ω values show distinct trajectories, demonstrating how the precision parameter controls the sensitivity of the system.

![k dependence](exponential_analysis_output/k_dependence.png)

### Parameter Space Heatmap
Filled contour heatmap over the (k, ω) grid with contour overlays. Useful for spotting ridges, valleys, and transitions across parameter regimes. The color intensity represents function values, while contour lines connect points of equal value, revealing the overall structure of the parameter space.

![Parameter heatmap](exponential_analysis_output/parameter_heatmap.png)

### 3D Surface
3D surface of the exponential term over the (k, ω) parameter space. Complements the heatmap with geometric intuition of gradients and curvature. The height represents function values, making it easy to visualize peaks, valleys, and the overall "landscape" of the function.

![3D surface](exponential_analysis_output/3d_surface.png)

## Key Insights

- **Low k regime**: Function behaves linearly (f ≈ 2ω)
- **High k regime**: Sharp, nonlinear transitions emerge
- **Symmetry**: Function exhibits symmetrical properties around ω = 0.5
- **Control parameter**: k acts as a "tuning dial" for system sensitivity

## Saved Arrays
Numerical grids are saved alongside the figures for reproducibility and downstream analysis:
- `exponential_analysis_output/omega_vec.npy` - Omega parameter values
- `exponential_analysis_output/k_vec.npy` - Precision parameter values
- `exponential_analysis_output/evaluation_grid.npy` - Computed function values
