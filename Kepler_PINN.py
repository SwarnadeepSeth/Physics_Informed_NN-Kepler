#!/usr/bin/env python3
"""
Kepler Orbit: Standard NN  |  Improved PINN  |  Hamiltonian NN (HNN)
======================================================================
Run a single method or all three:

  python3 Kepler_PINN.py                        # all three (default)
  python3 Kepler_PINN.py --method nn
  python3 Kepler_PINN.py --method pinn
  python3 Kepler_PINN.py --method hnn
  python3 Kepler_PINN.py --method all --orbits 3

Governing equations (dimensionless, GM=1):
    x_tt = -x/r^3,  y_tt = -y/r^3,  r = sqrt(x^2+y^2)

True Hamiltonian:
    H(x,y,px,py) = 0.5*(px^2+py^2) - GM/r

PINN IC fix (hard enforcement):
    xy(tau) = ic_pos + ic_vel_norm * tau + tau^2 * net(tau)
    guarantees xy(0)=ic_pos  AND  dxy/dtau(0)=ic_vel_norm  exactly,
    preventing convergence to a shifted-energy (larger) ellipse.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint, solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

os.makedirs("plots", exist_ok=True)
torch.set_default_dtype(torch.float32)

# Orbital constants (module-level so all helpers can access them)
GM = 1.0
a  = 1.0
e  = 0.5
x0  = a * (1.0 - e)
y0  = 0.0
vx0 = 0.0
vy0 = float(np.sqrt(GM * (1.0 + e) / (a * (1.0 - e))))
T_orb = float(2.0 * np.pi * np.sqrt(a**3 / GM))

COLORS = {"Exact": "#2ecc71", "NN": "#3498db", "PINN": "#e74c3c", "HNN": "#9b59b6"}


# ===========================================================================
# 0.  CLI ARGUMENTS
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Kepler orbit: NN / PINN / HNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method", choices=["nn", "pinn", "hnn", "all"], default="all",
        help="Which method to train and evaluate",
    )
    p.add_argument(
        "--orbits", type=float, default=1.5,
        help="Number of orbital periods to simulate",
    )
    p.add_argument(
        "--train-frac", type=float, default=None,
        help="Fraction of simulation used as training data "
             "(default: 0.5/orbits — exactly half the orbit, "
             "periapsis to apoapsis)",
    )
    return p.parse_args()


# ===========================================================================
# 1.  ORBITAL MECHANICS & REFERENCE SOLUTION
# ===========================================================================
def make_setup(n_orbits: float = 1.5, train_frac: float = 0.4,
               n_dense: int = 1000) -> dict:
    """Compute reference orbit and build all tensors needed by the runners."""
    T_SIM = n_orbits * T_orb

    def kepler_ode(state, t):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        return [vx, vy, -GM * x / r**3, -GM * y / r**3]

    t_dense = np.linspace(0.0, T_SIM, n_dense)
    sol = odeint(kepler_ode, [x0, y0, vx0, vy0], t_dense, rtol=1e-11, atol=1e-13)
    x_ref, y_ref, vx_ref, vy_ref = sol.T

    r_ref  = np.sqrt(x_ref**2 + y_ref**2)
    ax_ref = -GM * x_ref / r_ref**3
    ay_ref = -GM * y_ref / r_ref**3
    E_ref  = 0.5 * (vx_ref**2 + vy_ref**2) - GM / r_ref
    L_ref  = x_ref * vy_ref - y_ref * vx_ref

    # Normalise time to [0, 1] — critical for stable 2nd-order autograd
    t_norm   = t_dense / T_SIM
    t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1)
    xy_full  = torch.tensor(
        np.column_stack([x_ref, y_ref]), dtype=torch.float32
    )

    # Sparse training data — same points for all methods
    train_mask = t_dense <= train_frac * T_SIM
    train_idx  = np.where(train_mask)[0][::15]
    t_data     = t_tensor[train_idx]
    xy_data    = xy_full[train_idx]

    # Phase-space tensors for HNN: z=(x,y,vx,vy), dz/dt=(vx,vy,ax,ay)
    z_train = torch.tensor(
        np.column_stack([x_ref[train_idx], y_ref[train_idx],
                         vx_ref[train_idx], vy_ref[train_idx]]),
        dtype=torch.float32,
    )
    dz_train = torch.tensor(
        np.column_stack([vx_ref[train_idx], vy_ref[train_idx],
                         ax_ref[train_idx], ay_ref[train_idx]]),
        dtype=torch.float32,
    )

    SPLIT = int(train_frac * n_dense)

    print(f"\n  T_orb={T_orb:.4f}  T_sim={T_SIM:.4f} ({n_orbits} orbits)")
    print(f"  IC: x0={x0:.3f}, vy0={vy0:.4f}")
    print(f"  Training points: {len(train_idx)} / {n_dense}  "
          f"(t <= {train_frac * T_SIM:.3f})")
    print(f"  Reference energy range: {E_ref.ptp():.2e}  "
          f"L range: {L_ref.ptp():.2e}")

    return dict(
        T_SIM=T_SIM, T_orb=T_orb, n_dense=n_dense, train_frac=train_frac,
        t_dense=t_dense, t_norm=t_norm,
        x_ref=x_ref, y_ref=y_ref, vx_ref=vx_ref, vy_ref=vy_ref,
        ax_ref=ax_ref, ay_ref=ay_ref, E_ref=E_ref, L_ref=L_ref,
        t_tensor=t_tensor, xy_full=xy_full,
        t_data=t_data, xy_data=xy_data,
        z_train=z_train, dz_train=dz_train,
        SPLIT=SPLIT,
    )


# ===========================================================================
# 2.  ARCHITECTURES
# ===========================================================================
class FCN(nn.Module):
    """Fully-connected: normalised_time -> (x, y).  Baseline for NN."""
    def __init__(self, n_hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


class FCN_IC(nn.Module):
    """
    FCN with HARD initial-condition enforcement for PINN.

    The output is constructed so that:
        xy(tau=0)         = ic_pos       (position IC — exact to machine precision)
        d(xy)/d(tau)|_0   = ic_vel_norm  (velocity IC — exact)

    Transformation:
        xy(tau) = ic_pos + ic_vel_norm * tau + tau^2 * net(tau)

    Proof:
        xy(0)       = ic_pos + 0 + 0              = ic_pos          ✓
        xy'(tau)    = ic_vel_norm + 2*tau*net + tau^2*net'
        xy'(0)      = ic_vel_norm                                    ✓

    This eliminates soft IC loss terms and prevents the PINN from settling
    on a larger (shifted-energy) ellipse due to imprecise initial conditions.
    """
    def __init__(self, ic_pos: torch.Tensor, ic_vel_norm: torch.Tensor,
                 n_hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 2))
        self.net = nn.Sequential(*layers)
        # ICs as non-trained buffers
        self.register_buffer("ic_pos",      ic_pos.float().view(1, 2))
        self.register_buffer("ic_vel_norm", ic_vel_norm.float().view(1, 2))

    def forward(self, tau):
        return self.ic_pos + self.ic_vel_norm * tau + tau**2 * self.net(tau)


class HamiltonianNN(nn.Module):
    """
    Hamiltonian Neural Network (Greydanus, Dzamba, Yosinski — NeurIPS 2019).

    Learns a scalar H : R^4 -> R from phase-space training data.
    Hamilton's equations give the dynamics:
        dq/dt =  dH/dp   =>  dx/dt  =  dH/dpx,   dy/dt  =  dH/dpy
        dp/dt = -dH/dq   =>  dpx/dt = -dH/dx,    dpy/dt = -dH/dy
    The learned H is EXACTLY conserved along any HNN-generated trajectory
    (by antisymmetry of the symplectic matrix J).
    """
    def __init__(self, n_hidden: int = 128, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(4, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

    def time_derivative(self, z):
        """Compute dz/dt = J * grad_H via autograd.  z: (N, 4)."""
        z_r = z.detach().requires_grad_(True)
        H   = self.net(z_r).sum()
        dH  = torch.autograd.grad(H, z_r, create_graph=True)[0]
        dqdt =  dH[:, 2:4]   # dq/dt =  dH/dp
        dpdt = -dH[:, 0:2]   # dp/dt = -dH/dq
        return torch.cat([dqdt, dpdt], dim=1)


# ===========================================================================
# 3.  HELPERS
# ===========================================================================
def save_gif(outfile: str, files: list, fps: int = 10):
    imgs = [Image.open(f) for f in files]
    imgs[0].save(fp=outfile, format="GIF", append_images=imgs[1:],
                 save_all=True, duration=int(1000 / fps), loop=0)


def plot_frame(t_np, xy_ref_np, xy_pred_np, t_data_np, xy_data_np,
               train_frac, epoch, label, color, filename):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

    a1.plot(xy_ref_np[:, 0], xy_ref_np[:, 1], "#2ecc71", lw=2,
            label="Exact", zorder=3)
    a1.plot(xy_pred_np[:, 0], xy_pred_np[:, 1], color=color, lw=2.5,
            ls="--", alpha=0.85, label=f"{label} pred.", zorder=4)
    a1.scatter(xy_data_np[:, 0], xy_data_np[:, 1], s=60, c="#f39c12",
               zorder=5, edgecolors="k", lw=0.5, label="Train data")
    a1.scatter([0], [0], s=200, c="gold", marker="*", zorder=6,
               edgecolors="orange", lw=1, label="Star")
    a1.set_aspect("equal")
    a1.set_title(f"Orbit — epoch {epoch:,}", fontweight="bold")
    a1.set_xlabel("x"); a1.set_ylabel("y"); a1.legend(fontsize=9)

    a2.plot(t_np, xy_ref_np[:, 0], "#2ecc71", lw=1.8, label="x ref")
    a2.plot(t_np, xy_ref_np[:, 1], "#27ae60", lw=1.8, ls="--", label="y ref")
    a2.plot(t_np, xy_pred_np[:, 0], color=color, lw=2.5, label=f"x {label}")
    a2.plot(t_np, xy_pred_np[:, 1], color=color, lw=2.5, ls="--",
            label=f"y {label}")
    a2.axvspan(0, train_frac, alpha=0.08, color="orange", label="Train region")
    a2.set_xlabel("t (normalised)"); a2.set_ylabel("Position")
    a2.set_title("Position vs Time", fontweight="bold")
    a2.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(filename, dpi=90, bbox_inches="tight", facecolor="white")
    plt.close()


def conservation_laws(x, y, vx, vy):
    """Return (energy, angular_momentum) arrays."""
    r = np.sqrt(x**2 + y**2) + 1e-12
    E = 0.5 * (vx**2 + vy**2) - GM / r
    L = x * vy - y * vx
    return E, L


def _print_errors(name: str, err: np.ndarray, split: int):
    tr = float(np.nanmean(err[:split]))
    ex = float(np.nanmean(err[split:]))
    print(f"  {name:18s}  train MAE={tr:.5f}   extrapol MAE={ex:.5f}")


# ===========================================================================
# 4.  METHOD: STANDARD NN (data-only baseline)
# ===========================================================================
def run_nn(s: dict) -> dict:
    """Train the standard NN (data-only baseline) and return results dict."""
    print("\n" + "=" * 65)
    print("  [NN]  Standard Neural Network  (data-only baseline)")
    print("=" * 65)

    EPOCHS = 5_000
    torch.manual_seed(42)
    model = FCN()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses, frames = [], []

    for i in range(EPOCHS):
        optim.zero_grad()
        loss = torch.mean((model(s["t_data"]) - s["xy_data"]) ** 2)
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if (i + 1) % 250 == 0:
            with torch.no_grad():
                xy_pred = model(s["t_tensor"]).numpy()
            fn = f"plots/nn_{i+1:07d}.png"
            plot_frame(s["t_norm"], s["xy_full"].numpy(), xy_pred,
                       s["t_data"].numpy(), s["xy_data"].numpy(),
                       s["train_frac"], i + 1, "NN", COLORS["NN"], fn)
            frames.append(fn)
            if (i + 1) % 2500 == 0:
                print(f"    epoch {i+1:5d}  loss={loss.item():.3e}")

    save_gif("nn_kepler.gif", frames, fps=8)
    print("  ✓ nn_kepler.gif saved")

    with torch.no_grad():
        xy_nn = model(s["t_tensor"]).numpy()

    dt = s["t_dense"][1] - s["t_dense"][0]
    vx_fd = np.gradient(xy_nn[:, 0], dt)
    vy_fd = np.gradient(xy_nn[:, 1], dt)
    E_nn, L_nn = conservation_laws(xy_nn[:, 0], xy_nn[:, 1], vx_fd, vy_fd)
    err_nn = np.sqrt((xy_nn[:, 0] - s["x_ref"]) ** 2 +
                     (xy_nn[:, 1] - s["y_ref"]) ** 2)
    _print_errors("Standard NN", err_nn, s["SPLIT"])

    return dict(xy=xy_nn, E=E_nn, L=L_nn, err=err_nn,
                losses=losses, label="NN", color=COLORS["NN"], model=model)


# ===========================================================================
# 5.  METHOD: IMPROVED PINN  (hard IC + ODE residual)
# ===========================================================================
def run_pinn(s: dict) -> dict:
    """
    Train the improved PINN with hard-encoded initial conditions.

    Key improvements over naive PINN
    ---------------------------------
    * Hard IC via output transform — exact IC satisfaction, no IC loss terms
    * N_COLLOC = 500       (was 50)
    * LAMBDA_PHYS = 1.0    (was 1e-3)
    * Time normalised to [0, 1]  for stable 2nd-order autograd
    * StepLR decay (halved every 10 k epochs)
    """
    print("\n" + "=" * 65)
    print("  [PINN]  Improved PINN  (hard IC + ODE residual)")
    print("=" * 65)
    print("    N_COLLOC   = 500    (was 50)")
    print("    LAMBDA_PHYS= 1.0    (was 1e-3)")
    print("    IC enforcement: hard output transform (was soft loss)")

    T_SIM_t     = torch.tensor(s["T_SIM"], dtype=torch.float32)
    ic_pos      = torch.tensor([x0, y0], dtype=torch.float32)
    # In normalised time tau=t/T_SIM: dx/dtau = T_SIM * dx/dt
    ic_vel_norm = torch.tensor([vx0 * s["T_SIM"], vy0 * s["T_SIM"]],
                               dtype=torch.float32)

    N_COLLOC    = 500
    LAMBDA_PHYS = 1.0
    EPOCHS      = 30_000
    LR          = 5e-4

    t_coll = torch.linspace(0.0, 1.0, N_COLLOC).view(-1, 1).requires_grad_(True)

    torch.manual_seed(42)
    model = FCN_IC(ic_pos, ic_vel_norm)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10_000, gamma=0.5)

    loss_total_h, loss_data_h, loss_phys_h = [], [], []
    frames = []

    for i in range(EPOCHS):
        optim.zero_grad()

        # data loss
        loss_d = torch.mean((model(s["t_data"]) - s["xy_data"]) ** 2)

        # physics (ODE residual) loss in normalised time
        pred_p = model(t_coll)
        xp = pred_p[:, 0:1]
        yp = pred_p[:, 1:2]

        vx_p = torch.autograd.grad(
            xp, t_coll, torch.ones_like(xp), create_graph=True)[0]
        vy_p = torch.autograd.grad(
            yp, t_coll, torch.ones_like(yp), create_graph=True)[0]
        ax_p = torch.autograd.grad(
            vx_p, t_coll, torch.ones_like(vx_p), create_graph=True)[0]
        ay_p = torch.autograd.grad(
            vy_p, t_coll, torch.ones_like(vy_p), create_graph=True)[0]

        # chain rule: d²x/dtau² = T_SIM² * d²x/dt² = T_SIM² * (-GM*x/r³)
        r_p   = torch.sqrt(xp**2 + yp**2 + 1e-8)
        res_x = ax_p - T_SIM_t**2 * (-GM * xp / r_p**3)
        res_y = ay_p - T_SIM_t**2 * (-GM * yp / r_p**3)
        loss_p = torch.mean(res_x**2 + res_y**2)

        loss = loss_d + LAMBDA_PHYS * loss_p
        loss.backward()
        optim.step()
        sched.step()

        loss_total_h.append(loss.item())
        loss_data_h.append(loss_d.item())
        loss_phys_h.append(loss_p.item())

        if (i + 1) % 1000 == 0:
            with torch.no_grad():
                xy_pred = model(s["t_tensor"]).numpy()
            fn = f"plots/pinn_{i+1:07d}.png"
            plot_frame(s["t_norm"], s["xy_full"].numpy(), xy_pred,
                       s["t_data"].numpy(), s["xy_data"].numpy(),
                       s["train_frac"], i + 1, "PINN", COLORS["PINN"], fn)
            frames.append(fn)
            if (i + 1) % 5000 == 0:
                print(f"    epoch {i+1:6d} | total={loss.item():.3e} "
                      f"data={loss_d.item():.3e} phys={loss_p.item():.3e}")

    save_gif("pinn_kepler.gif", frames, fps=8)
    print("  ✓ pinn_kepler.gif saved")

    with torch.no_grad():
        xy_pinn = model(s["t_tensor"]).numpy()

    dt = s["t_dense"][1] - s["t_dense"][0]
    vx_fd = np.gradient(xy_pinn[:, 0], dt)
    vy_fd = np.gradient(xy_pinn[:, 1], dt)
    E_pinn, L_pinn = conservation_laws(xy_pinn[:, 0], xy_pinn[:, 1], vx_fd, vy_fd)
    err_pinn = np.sqrt((xy_pinn[:, 0] - s["x_ref"]) ** 2 +
                       (xy_pinn[:, 1] - s["y_ref"]) ** 2)
    _print_errors("Improved PINN", err_pinn, s["SPLIT"])

    return dict(xy=xy_pinn, E=E_pinn, L=L_pinn, err=err_pinn,
                losses=loss_total_h, data_losses=loss_data_h,
                phys_losses=loss_phys_h, LAMBDA_PHYS=LAMBDA_PHYS,
                label="PINN", color=COLORS["PINN"], model=model)


# ===========================================================================
# 6.  METHOD: HAMILTONIAN NN
# ===========================================================================
def _integrate_hnn(model: HamiltonianNN, T_SIM: float,
                   t_eval: np.ndarray) -> np.ndarray:
    """
    Integrate the HNN trajectory via scipy.solve_ivp (RK45).
    Returns a (4, N) array: rows are [x, y, vx, vy].
    Pads with NaN if integration does not reach t_eval[-1].
    """
    def hnn_rhs(t, state):
        with torch.enable_grad():
            z  = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            z  = z.requires_grad_(True)
            H  = model(z).sum()
            dH = torch.autograd.grad(H, z)[0].squeeze()
        return [dH[2].item(), dH[3].item(), -dH[0].item(), -dH[1].item()]

    sol = solve_ivp(
        hnn_rhs, t_span=[0.0, T_SIM], y0=[x0, y0, vx0, vy0],
        t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8,
    )
    n_ref = len(t_eval)
    if sol.success and sol.y.shape[1] == n_ref:
        return sol.y
    n = sol.y.shape[1]
    pad = lambda a: np.pad(a, (0, n_ref - n), constant_values=np.nan)
    return np.vstack([pad(sol.y[i]) for i in range(4)])


def run_hnn(s: dict) -> dict:
    """
    Train the Hamiltonian Neural Network and return results dict.
    Generates hnn_kepler.gif by integrating the HNN at training checkpoints.
    """
    print("\n" + "=" * 65)
    print("  [HNN]  Hamiltonian Neural Network  (Greydanus et al. 2019)")
    print("=" * 65)
    print("  Input : z = (x, y, px, py)")
    print("  Output: scalar H(z)  [learned Hamiltonian]")
    print("  GIF   : integration snapshot every 500 epochs")

    EPOCHS       = 10_000
    GIF_INTERVAL = 500

    torch.manual_seed(42)
    model = HamiltonianNN(n_hidden=128, n_layers=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=EPOCHS, eta_min=1e-5)

    losses, frames = [], []

    for i in range(EPOCHS):
        optim.zero_grad()
        dz_pred = model.time_derivative(s["z_train"])
        loss    = torch.mean((dz_pred - s["dz_train"]) ** 2)
        loss.backward()
        optim.step()
        sched.step()
        losses.append(loss.item())

        if (i + 1) % GIF_INTERVAL == 0:
            # Integrate current HNN snapshot and save frame
            y_sol = _integrate_hnn(model, s["T_SIM"], s["t_dense"])
            xy_snap = np.column_stack([y_sol[0], y_sol[1]])
            fn = f"plots/hnn_{i+1:07d}.png"
            plot_frame(s["t_norm"], s["xy_full"].numpy(), xy_snap,
                       s["t_data"].numpy(), s["xy_data"].numpy(),
                       s["train_frac"], i + 1, "HNN", COLORS["HNN"], fn)
            frames.append(fn)
            if (i + 1) % 2000 == 0:
                print(f"    epoch {i+1:6d}  loss={loss.item():.3e}")

    save_gif("hnn_kepler.gif", frames, fps=8)
    print("  ✓ hnn_kepler.gif saved")

    print("  Integrating final HNN trajectory... ", end="", flush=True)
    y_sol = _integrate_hnn(model, s["T_SIM"], s["t_dense"])
    x_hnn, y_hnn, vx_hnn, vy_hnn = y_sol
    print("done")

    E_hnn, L_hnn = conservation_laws(x_hnn, y_hnn, vx_hnn, vy_hnn)

    z_traj = torch.tensor(
        np.column_stack([x_hnn, y_hnn, vx_hnn, vy_hnn]), dtype=torch.float32)
    with torch.no_grad():
        H_learned = model(z_traj).numpy().flatten()

    err_hnn = np.sqrt((x_hnn - s["x_ref"]) ** 2 + (y_hnn - s["y_ref"]) ** 2)
    _print_errors("HNN", err_hnn, s["SPLIT"])

    return dict(xy=np.column_stack([x_hnn, y_hnn]),
                vx=vx_hnn, vy=vy_hnn,
                E=E_hnn, L=L_hnn, H_learned=H_learned,
                err=err_hnn, losses=losses,
                label="HNN", color=COLORS["HNN"], model=model)


# ===========================================================================
# 7.  COMPARISON FIGURE  (8 panels — only built when running all three)
# ===========================================================================
def run_comparison(results: dict, s: dict):
    """Build the 8-panel comparison figure and print summary table."""
    fig = plt.figure(figsize=(18, 15))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_orb  = fig.add_subplot(gs[0, :2])
    ax_xt   = fig.add_subplot(gs[0, 2])
    ax_err  = fig.add_subplot(gs[1, 0])
    ax_E    = fig.add_subplot(gs[1, 1])
    ax_L    = fig.add_subplot(gs[1, 2])
    ax_loss = fig.add_subplot(gs[2, 0])
    ax_Hl   = fig.add_subplot(gs[2, 1])
    ax_Hnn  = fig.add_subplot(gs[2, 2])

    # (A) Trajectories
    ax_orb.plot(s["x_ref"], s["y_ref"], COLORS["Exact"], lw=3,
                label="Exact", zorder=3)
    for key, ls in [("nn", "-."), ("pinn", "--"), ("hnn", ":")]:
        if key in results:
            r = results[key]
            ax_orb.plot(r["xy"][:, 0], r["xy"][:, 1],
                        COLORS[r["label"]], lw=2, ls=ls,
                        alpha=0.9, label=r["label"])
    ax_orb.scatter(s["xy_data"].numpy()[:, 0], s["xy_data"].numpy()[:, 1],
                   s=70, c="#f39c12", zorder=5, edgecolors="k", lw=0.5,
                   label="Training data")
    ax_orb.scatter([0], [0], s=300, c="gold", marker="*", zorder=6,
                   edgecolors="darkorange", lw=1.5, label="Star")
    ax_orb.set_aspect("equal")
    ax_orb.set_title("(A) Orbital Trajectories", fontweight="bold", fontsize=12)
    ax_orb.set_xlabel("x"); ax_orb.set_ylabel("y")
    ax_orb.legend(fontsize=9, ncol=2, framealpha=0.9)

    # (B) x(t)
    ax_xt.plot(s["t_dense"], s["x_ref"], COLORS["Exact"], lw=2, label="Exact")
    for key, ls in [("nn", "-."), ("pinn", "--"), ("hnn", ":")]:
        if key in results:
            r = results[key]
            ax_xt.plot(s["t_dense"], r["xy"][:, 0],
                       COLORS[r["label"]], lw=1.5, ls=ls, label=r["label"])
    ax_xt.axvspan(0, s["train_frac"] * s["T_SIM"], alpha=0.08, color="orange")
    ax_xt.set_title("(B) x(t)", fontweight="bold", fontsize=12)
    ax_xt.set_xlabel("Time (s)"); ax_xt.set_ylabel("x")
    ax_xt.legend(fontsize=8); ax_xt.grid(alpha=0.3)

    # (C) Position error (log)
    for key in ["nn", "pinn", "hnn"]:
        if key in results:
            r = results[key]
            ax_err.semilogy(s["t_dense"], r["err"] + 1e-10,
                            COLORS[r["label"]], lw=2, label=r["label"])
    ax_err.axvline(s["train_frac"] * s["T_SIM"], color="gray",
                   ls=":", lw=1.5, label="Train end")
    ax_err.set_title("(C) Position Error |Δr|", fontweight="bold", fontsize=12)
    ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("|Δr| (log)")
    ax_err.legend(fontsize=9); ax_err.grid(alpha=0.3)

    # (D) Energy conservation
    ax_E.plot(s["t_dense"], s["E_ref"], COLORS["Exact"], lw=2.5,
              label="Exact E")
    for key in ["nn", "pinn", "hnn"]:
        if key in results:
            r = results[key]
            ax_E.plot(s["t_dense"], r["E"], COLORS[r["label"]],
                      lw=1.5, ls="--", alpha=0.85, label=r["label"])
    ax_E.set_title("(D) Energy E(t)", fontweight="bold", fontsize=12)
    ax_E.set_xlabel("Time"); ax_E.set_ylabel("Energy")
    ax_E.legend(fontsize=8); ax_E.grid(alpha=0.3)

    # (E) Angular momentum
    ax_L.plot(s["t_dense"], s["L_ref"], COLORS["Exact"], lw=2.5,
              label="Exact L")
    for key in ["nn", "pinn", "hnn"]:
        if key in results:
            r = results[key]
            ax_L.plot(s["t_dense"], r["L"], COLORS[r["label"]],
                      lw=1.5, ls="--", alpha=0.85, label=r["label"])
    ax_L.set_title("(E) Angular Momentum L(t)", fontweight="bold", fontsize=12)
    ax_L.set_xlabel("Time"); ax_L.set_ylabel("L")
    ax_L.legend(fontsize=8); ax_L.grid(alpha=0.3)

    # (F) Loss curves
    if "nn" in results:
        ax_loss.semilogy(results["nn"]["losses"], COLORS["NN"],
                         lw=1.5, label="NN data loss")
    if "pinn" in results:
        lp = results["pinn"]["LAMBDA_PHYS"]
        ax_loss.semilogy(results["pinn"]["data_losses"], COLORS["PINN"],
                         lw=1.5, label="PINN data loss")
        ax_loss.semilogy([lp * v for v in results["pinn"]["phys_losses"]],
                         "#e74c3c", lw=1, ls=":", alpha=0.6,
                         label="λ × PINN phys")
    if "hnn" in results:
        ax_loss.semilogy(results["hnn"]["losses"], COLORS["HNN"],
                         lw=1.5, label="HNN loss")
    ax_loss.set_title("(F) Training Loss Curves", fontweight="bold", fontsize=12)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss (log)")
    ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)

    # (G) Learned H vs true energy
    ax_Hl.plot(s["t_dense"], s["E_ref"], COLORS["Exact"], lw=2.5,
               label="True energy E")
    if "hnn" in results:
        ax_Hl.plot(s["t_dense"], results["hnn"]["H_learned"],
                   COLORS["HNN"], lw=2, ls="--", label="Learned H (HNN)")
    ax_Hl.set_title("(G) HNN Learned H vs True Energy",
                    fontweight="bold", fontsize=12)
    ax_Hl.set_xlabel("Time"); ax_Hl.set_ylabel("Value")
    ax_Hl.legend(fontsize=9); ax_Hl.grid(alpha=0.3)

    # (H) HNN orbit detail
    ax_Hnn.plot(s["x_ref"], s["y_ref"], COLORS["Exact"], lw=3, label="Exact")
    if "hnn" in results:
        ax_Hnn.plot(results["hnn"]["xy"][:, 0], results["hnn"]["xy"][:, 1],
                    COLORS["HNN"], lw=2.5, ls="--", alpha=0.9, label="HNN")
    ax_Hnn.scatter([0], [0], s=200, c="gold", marker="*", zorder=6,
                   edgecolors="darkorange", lw=1.5)
    ax_Hnn.set_aspect("equal")
    ax_Hnn.set_title("(H) HNN Orbit Detail", fontweight="bold", fontsize=12)
    ax_Hnn.set_xlabel("x"); ax_Hnn.set_ylabel("y")
    ax_Hnn.legend(fontsize=9)

    fig.suptitle(
        f"Kepler Orbit: NN vs PINN vs HNN  |  e={e}, GM={GM}, "
        f"{s['T_SIM'] / T_orb:.1f} orbits",
        fontsize=15, fontweight="bold",
    )
    plt.savefig("plots/kepler_comparison.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("\n  ✓ plots/kepler_comparison.png saved")

    # Summary table
    print("\n" + "=" * 79)
    print(f"  {'Method':<18} {'Train MAE':>12} {'Extrapol MAE':>14} "
          f"{'|ΔE| extrapol':>15} {'|ΔL| extrapol':>15}")
    print("=" * 79)
    for key in ["nn", "pinn", "hnn"]:
        if key not in results:
            continue
        r = results[key]
        tr = float(np.nanmean(r["err"][:s["SPLIT"]]))
        ex = float(np.nanmean(r["err"][s["SPLIT"]:]))
        dE = float(np.nanmean(np.abs(r["E"][s["SPLIT"]:] -
                                     s["E_ref"][s["SPLIT"]:])))
        dL = float(np.nanmean(np.abs(r["L"][s["SPLIT"]:] -
                                     s["L_ref"][s["SPLIT"]:])))
        print(f"  {r['label']:<18} {tr:>12.5f} {ex:>14.5f} "
              f"{dE:>15.6f} {dL:>15.6f}")
    print("=" * 79)
    print(f"\n  Exact: E = {s['E_ref'][0]:.6f}  "
          f"(analytical: -GM/2a = {-GM/(2*a):.6f})")
    print(f"  Exact: L = {s['L_ref'][0]:.6f}  "
          f"(analytical: sqrt(GM*a*(1-e²)) = "
          f"{float(np.sqrt(GM * a * (1 - e**2))):.6f})")
    print("\n  HNN : learned H is EXACTLY conserved along its trajectory.")
    print("  PINN: conservation emerges from minimised ODE residual.")
    print("  NN  : no conservation mechanism — energy drift expected.")
    print("=" * 79)


# ===========================================================================
# 8.  MAIN
# ===========================================================================
def main():
    args = parse_args()

    # Default training fraction = exactly half the orbit (periapsis → apoapsis)
    train_frac = args.train_frac if args.train_frac is not None \
        else 0.5 / args.orbits

    print("=" * 65)
    print("  Kepler Orbit  —  NN vs Improved PINN vs HNN")
    print("=" * 65)
    print(f"  Method     : {args.method}")
    print(f"  Orbits     : {args.orbits}")
    print(f"  Train data : first half-orbit (train_frac={train_frac:.4f})")
    print(f"  e={e}, GM={GM}, a={a},  T_orb={T_orb:.4f}")

    s = make_setup(n_orbits=args.orbits, train_frac=train_frac)

    results = {}
    if args.method in ("nn",   "all"):
        results["nn"]   = run_nn(s)
    if args.method in ("pinn", "all"):
        results["pinn"] = run_pinn(s)
    if args.method in ("hnn",  "all"):
        results["hnn"]  = run_hnn(s)

    if args.method == "all":
        run_comparison(results, s)


if __name__ == "__main__":
    main()
