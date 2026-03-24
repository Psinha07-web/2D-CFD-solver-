A complete pipeline for simulating 2D incompressible fluid flow with a classical
Navier-Stokes solver, then training a Fourier Neural Operator (FNO) to predict
future flow states at a fraction of the compute cost.

---

## Project structure

```
cfd_fno/
├── main.py               # End-to-end pipeline entry point
├── train.py              # FNO training loop + checkpointing
├── visualize.py          # Velocity / pressure / vorticity plots + animations
├── requirements.txt
├── solver/
│   └── cfd_solver.py     # 2D Navier-Stokes solver (projection method)
└── models/
    ├── fno2d.py           # Fourier Neural Operator architecture
    └── dataset.py         # PyTorch Dataset + physics-aware loss
```

---

## Installation

```bash
pip install -r requirements.txt
```

CUDA is optional but strongly recommended for training.

---

## Quick start

### 1. Generate CFD training data

```bash
python main.py --mode generate
```

Runs the lid-driven cavity solver at Re=400 for 2000 timesteps, saving snapshots
of u, v, p, vorticity, and speed to `data/snapshots.h5`.
Also produces:
- `outputs/cfd_snapshot.png`   — 4-panel dashboard (velocity, streamlines, pressure, vorticity)
- `outputs/velocity_animation.gif`
- `outputs/vorticity_animation.gif`

### 2. Train the FNO

```bash
python main.py --mode train
```

Trains FNO2D to predict (u, v, p) at t+1 from (u, v, p) at t.
Saves checkpoints to `checkpoints/` and training history plot to `outputs/`.

### 3. Run inference and compare against CFD

```bash
python main.py --mode infer
```

Rolls out the trained FNO for 50 steps and compares against ground truth CFD.
Produces:
- `outputs/comparison_t*.png`     — side-by-side CFD vs FNO vs error
- `outputs/fno_prediction.gif`    — animated FNO rollout
- `outputs/rollout_error.png`     — per-step relative L2 error curve

### 4. Run everything in sequence

```bash
python main.py --mode all
```

---

## Configuration

Edit `DEFAULT_CFG` in `main.py` or pass a JSON config:

```bash
python main.py --mode all --config my_config.json
```

Key parameters:

| Parameter         | Default | Description                              |
|-------------------|---------|------------------------------------------|
| `Nx`, `Ny`        | 64      | Grid resolution                          |
| `Re`              | 400     | Reynolds number                          |
| `n_steps`         | 2000    | Solver timesteps per trajectory          |
| `snapshot_every`  | 5       | Save snapshot every N steps              |
| `n_trajectories`  | 1       | Number of Re variants to generate        |
| `modes1/2`        | 12      | Fourier modes kept (higher = more detail)|
| `width`           | 32      | FNO latent channel width                 |
| `n_layers`        | 4       | Number of FNO blocks                     |
| `epochs`          | 100     | Training epochs                          |
| `lr`              | 1e-3    | Initial learning rate                    |
| `beta_div`        | 0.05    | Divergence penalty weight in loss        |

---

## Architecture overview

### CFD Solver (`solver/cfd_solver.py`)

Solves the 2D incompressible Navier-Stokes equations using:
- **Staggered MAC grid** (u on vertical faces, v on horizontal faces, p at centers)
- **Semi-Lagrangian advection** (unconditionally stable)
- **Explicit diffusion** (central differences)
- **Pressure projection** (Poisson equation via direct solve)

### Fourier Neural Operator (`models/fno2d.py`)

Each FNO block:
1. Applies a learned convolution in Fourier space (keeps low-frequency modes only)
2. Adds a pointwise bypass convolution (1×1 conv)
3. Applies GELU activation + instance norm

The full model lifts the input to a wider latent space, applies N FNO blocks,
then projects back to the output channels.

### Training (`train.py`)

- **Loss**: MSE + divergence-free penalty `||∇·u_pred||²`
- **Optimizer**: AdamW with cosine annealing LR schedule
- **Mixed precision** (AMP) for GPU speedup
- **Gradient clipping** for stability

---

## Extending the project

### Multiple Reynolds numbers
Set `n_trajectories > 1` to generate data at multiple Re values, giving the FNO
generalization across flow regimes.

### Higher resolution
Increase `Nx`, `Ny` (e.g. 128 or 256) and `modes1/2` (e.g. 16–24). Training will
require a GPU.

### Multi-step rollout training
Replace single-step MSE loss with an accumulated rollout loss (sum of errors over
K future steps). This dramatically improves long-horizon prediction stability.

### Different boundary conditions
Modify `apply_boundary_conditions` in `cfd_solver.py` to simulate:
- Channel flow (inflow/outflow)
- Flow around a cylinder (immersed boundary)
- Periodic domains (turbulence)

---

## Expected performance

On a 64×64 grid, Re=400, after 100 epochs:
- Relative L2 error (1-step): ~1–3%
- Rollout error grows slowly, staying under ~10% for ~20 steps
- FNO inference is ~100× faster than the CFD solver per timestep
