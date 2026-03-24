"""
main.py — End-to-end pipeline for 2D CFD + FNO

Usage:
  python main.py --mode generate          # Run CFD solver, save training data
  python main.py --mode train             # Train FNO on saved data
  python main.py --mode infer             # Run FNO inference and compare with CFD
  python main.py --mode all               # All of the above in sequence
"""

import argparse
import os
import json
import numpy as np
import torch


DEFAULT_CFG = {
    # Solver
    "Nx": 64,
    "Ny": 64,
    "Lx": 1.0,
    "Ly": 1.0,
    "Re": 400.0,
    "dt": 0.001,
    "n_steps": 2000,      
    "snapshot_every": 5,   
    "u_lid": 1.0,
    "n_trajectories": 1,    

    # Data
    "data_dir":    "data",
    "data_file":   "data/snapshots.h5",
    "fields":      ["u", "v", "p"],

    # FNO
    "in_channels":  3,
    "out_channels": 3,
    "modes1":       12,
    "modes2":       12,
    "width":        32,
    "n_layers":     4,

    # Training
    "epochs":          100,
    "batch_size":      16,
    "lr":              1e-3,
    "lr_min":          1e-5,
    "weight_decay":    1e-4,
    "grad_clip":       1.0,
    "alpha_mse":       1.0,
    "beta_div":        0.05,
    "use_amp":         True,
    "log_every":       20,
    "checkpoint_dir":  "checkpoints",
    "resume_checkpoint": None,

    # Inference
    "rollout_steps": 50,
    "output_dir":    "outputs",
}



def generate_data(cfg):
    from solver.cfd_solver import CFDSolver2D, save_snapshots_hdf5
    from visualize import plot_dashboard, animate_snapshots

    os.makedirs(cfg["data_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    Re_values = np.linspace(100, cfg["Re"], cfg["n_trajectories"])

    all_u, all_v, all_p, all_w, all_s = [], [], [], [], []

    for idx, Re in enumerate(Re_values):
        print(f"\n[{idx+1}/{cfg['n_trajectories']}] Solving Re={Re:.0f} ...")
        solver = CFDSolver2D(
            Nx=cfg["Nx"], Ny=cfg["Ny"],
            Lx=cfg["Lx"], Ly=cfg["Ly"],
            Re=Re, dt=cfg["dt"]
        )
        snaps = solver.run(
            n_steps=cfg["n_steps"],
            snapshot_every=cfg["snapshot_every"],
            u_lid=cfg["u_lid"]
        )
        all_u.append(snaps["u"])
        all_v.append(snaps["v"])
        all_p.append(snaps["p"])
        all_w.append(snaps["vorticity"])
        all_s.append(snaps["speed"])

  
    combined = {
        "u":        np.concatenate(all_u, axis=0),
        "v":        np.concatenate(all_v, axis=0),
        "p":        np.concatenate(all_p, axis=0),
        "vorticity": np.concatenate(all_w, axis=0),
        "speed":    np.concatenate(all_s, axis=0),
    }

    meta = {k: cfg[k] for k in ("Nx", "Ny", "Re", "dt", "n_steps", "snapshot_every")}
    save_snapshots_hdf5(combined, cfg["data_file"], metadata=meta)

    # Visualize last snapshot
    T = combined["u"].shape[0]
    print("\nSaving visualization of final snapshot ...")
    plot_dashboard(
        combined["u"][-1], combined["v"][-1],
        combined["p"][-1], combined["vorticity"][-1],
        title=f"CFD snapshot (t={T * cfg['snapshot_every'] * cfg['dt']:.2f}s)",
        save_path=os.path.join(cfg["output_dir"], "cfd_snapshot.png")
    )

    print("Saving animation ...")
    animate_snapshots(
        combined, field="speed", fps=15,
        save_path=os.path.join(cfg["output_dir"], "velocity_animation.gif")
    )

    animate_snapshots(
        combined, field="vorticity", fps=15,
        save_path=os.path.join(cfg["output_dir"], "vorticity_animation.gif")
    )

    print(f"\nData generation complete. Shapes:")
    for k, v in combined.items():
        print(f"  {k}: {v.shape}")



def train_model(cfg):
    from models.dataset import CFDDataset, make_dataloaders
    from train import build_trainer
    from visualize import plot_training_history

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    dataset = CFDDataset(
        data_path=cfg["data_file"],
        fields=cfg["fields"],
        normalize=True
    )
    train_loader, val_loader = make_dataloaders(
        dataset,
        batch_size=cfg["batch_size"]
    )

    # Attach grid spacing to cfg
    cfg["dx"] = cfg["Lx"] / cfg["Nx"]
    cfg["dy"] = cfg["Ly"] / cfg["Ny"]

    trainer = build_trainer(cfg, train_loader, val_loader, device=device)
    trainer.train(epochs=cfg["epochs"])

    os.makedirs(cfg["output_dir"], exist_ok=True)
    plot_training_history(
        trainer.history,
        save_path=os.path.join(cfg["output_dir"], "training_history.png")
    )




def run_inference(cfg):
    from models.dataset import CFDDataset
    from models.fno2d import FNO2DRollout
    from train import relative_l2_error
    from visualize import plot_comparison, animate_snapshots

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)

   
    dataset = CFDDataset(
        data_path=cfg["data_file"],
        fields=cfg["fields"],
        normalize=True
    )

    # Load best checkpoint
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "fno2d_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg["checkpoint_dir"], "fno2d_final.pt")
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt.get("cfg", cfg)
    model = FNO2DRollout(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 3),
        modes1=model_cfg.get("modes1", 12),
        modes2=model_cfg.get("modes2", 12),
        width=model_cfg.get("width", 32),
        n_layers=model_cfg.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded. Parameters: {model.count_parameters():,}")

   
    x0, y_gt = dataset[0]
    x0 = x0.unsqueeze(0).to(device)

    T = cfg["rollout_steps"]
    print(f"\nRolling out {T} steps ...")
    with torch.no_grad():
        pred_trajectory = model.rollout(x0, T)

    pred_trajectory = pred_trajectory.squeeze(0).cpu()  

    # Denormalize
    pred_phys = dataset.denormalize(pred_trajectory)

    gt_sequence = []
    xi = x0.cpu()
    for t in range(T):
        idx = min(t + 1, len(dataset) - 1)
        gt_sequence.append(dataset[idx][1]) 
    gt_tensor = torch.stack(gt_sequence, dim=0) 
    gt_phys = dataset.denormalize(gt_tensor)

    rel_errors = []
    for t in range(T):
        err = relative_l2_error(
            pred_trajectory[t].unsqueeze(0),
            gt_tensor[t].unsqueeze(0)
        )
        rel_errors.append(err)

    mean_err = np.mean(rel_errors)
    print(f"Mean relative L2 error over {T} steps: {mean_err:.4f}")

    for t_plot in [0, T // 4, T // 2, T - 1]:
        plot_comparison(
            gt_phys[t_plot].numpy(),
            pred_phys[t_plot].numpy(),
            field_names=cfg["fields"],
            step=t_plot,
            save_path=os.path.join(cfg["output_dir"], f"comparison_t{t_plot:03d}.png")
        )
    u_idx = cfg["fields"].index("u") if "u" in cfg["fields"] else 0
    v_idx = cfg["fields"].index("v") if "v" in cfg["fields"] else 1
    speed = torch.sqrt(pred_phys[:, u_idx] ** 2 + pred_phys[:, v_idx] ** 2).numpy()
    animate_snapshots(
        {"speed": speed}, field="speed", fps=12,
        save_path=os.path.join(cfg["output_dir"], "fno_prediction.gif")
    )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(T), rel_errors, marker="o", markersize=3)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("FNO rollout error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], "rollout_error.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAll inference outputs saved to: {cfg['output_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D CFD + FNO pipeline")
    parser.add_argument("--mode", choices=["generate", "train", "infer", "all"],
                        default="all")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file (overrides defaults)")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CFG)

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))
        print(f"Loaded config from {args.config}")

    if args.mode in ("generate", "all"):
        print("\n" + "=" * 60)
        print(" STEP 1: Generating CFD training data")
        print("=" * 60)
        generate_data(cfg)

    if args.mode in ("train", "all"):
        print("\n" + "=" * 60)
        print(" STEP 2: Training FNO")
        print("=" * 60)
        train_model(cfg)

    if args.mode in ("infer", "all"):
        print("\n" + "=" * 60)
        print(" STEP 3: FNO Inference")
        print("=" * 60)
        run_inference(cfg)
