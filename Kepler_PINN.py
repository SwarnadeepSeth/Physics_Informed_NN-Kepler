#!/usr/bin/env python3
"""
Kepler Orbit: Standard NN vs Improved PINN vs Hamiltonian NN (HNN)
====================================================================
Problem
-------
Reconstruct a Keplerian elliptical orbit from sparse position observations
covering the first 40% of the trajectory; extrapolate the remainder.

Governing equations (dimensionless, GM=1):
    x_tt = -x/r^3,  y_tt = -y/r^3,  r = sqrt(x^2+y^2)

True Hamiltonian (conserved mechanical energy):
    H(x,y,px,py) = 0.5*(px^2+py^2) - GM/r

Three methods compared on identical sparse training data
---------------------------------------------------------
1. Standard NN   - regression  t -> (x,y)                    [baseline]
2. Improved PINN - ODE residual + IC position/velocity loss   [physics-constrained]
3. HNN           - learns H(q,p), integrates Hamilton's eqs   [structure-preserving]

Improvements over naive PINN
------------------------------
 * N_COLLOC   : 50   -> 500
 * LAMBDA_PHYS: 1e-3 -> 1.0
 * IC position + velocity loss  (LAMBDA_IC = 10.0)
 * Time normalised to [0,1]  for stable 2nd-order autograd
 * StepLR decay schedule
 * Hamiltonian NN (Greydanus et al. NeurIPS 2019)

HNN principle
-------------
 - Architecture: z=(x,y,px,py) -> H(z) in R
 - Training:  MSE on Hamilton's equations at training phase-space points
              loss = MSE(dH/dpx - vx) + MSE(dH/dpy - vy)
                   + MSE(dH/dx  + ax) + MSE(dH/dy  + ay)
 - Inference: scipy.solve_ivp with dx/dt=dH/dpx, dpx/dt=-dH/dx, etc.
 - The LEARNED H is exactly conserved along the inferred trajectory
   Proof: d/dt H(z(t)) = grad_H . dz/dt = grad_H . J*grad_H = 0
          because J = [[0,I],[-I,0]] is antisymmetric.
"""

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

# =============================================================================
# 1.  ORBITAL MECHANICS SETUP
# =============================================================================
GM          = 1.0          # gravitational parameter (dimensionless)
a           = 1.0          # semi-major axis
e           = 0.5          # eccentricity  (0=circle, <1=ellipse)

x0  = a * (1.0 - e)                                    # periapsis distance
y0  = 0.0
vx0 = 0.0                                               # no radial vel. at periapsis
vy0 = np.sqrt(GM * (1.0 + e) / (a * (1.0 - e)))       # vis-viva equation

T_orb  = 2.0 * np.pi * np.sqrt(a**3 / GM)             # Kepler III
T_SIM  = 1.5 * T_orb                                   # simulate 1.5 orbits
N_DENSE = 1000                                          # dense reference grid
TRAIN_FRAC = 0.4                                        # fraction used for training

print("=" * 65)
print("  Kepler Orbit  --  NN vs Improved PINN vs HNN")
print("=" * 65)
print(f"  e={e}, GM={GM}, a={a}")
print(f"  T_orb={T_orb:.4f}, T_sim={T_SIM:.4f}")
print(f"  IC: x0={x0:.3f}, y0={y0}, vx0={vx0}, vy0={vy0:.4f}")
print("=" * 65)

# =============================================================================
# 2.  NUMERICAL REFERENCE  (high-accuracy odeint)
# =============================================================================
def kepler_ode(state, t):
    x, y, vx, vy = state
    r  = np.sqrt(x**2 + y**2)
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    return [vx, vy, ax, ay]

t_dense = np.linspace(0.0, T_SIM, N_DENSE)
sol     = odeint(kepler_ode, [x0, y0, vx0, vy0], t_dense, rtol=1e-11, atol=1e-13)
x_ref, y_ref, vx_ref, vy_ref = sol.T

r_ref  = np.sqrt(x_ref**2 + y_ref**2)
ax_ref = -GM * x_ref / r_ref**3
ay_ref = -GM * y_ref / r_ref**3
E_ref  = 0.5*(vx_ref**2 + vy_ref**2) - GM/r_ref   # energy (conserved)
L_ref  = x_ref*vy_ref - y_ref*vx_ref               # angular momentum (conserved)

print(f"\n  Reference conservation check (should be CONSTANT):")
print(f"    Energy    in [{E_ref.min():.8f}, {E_ref.max():.8f}]  "
      f"range={E_ref.ptp():.2e}")
print(f"    Ang.mom   in [{L_ref.min():.8f}, {L_ref.max():.8f}]  "
      f"range={L_ref.ptp():.2e}")

# =============================================================================
# 3.  TENSORS
# =============================================================================
# Normalise time to [0,1] -- critical for stable 2nd-order autograd in PINN
t_norm   = t_dense / T_SIM
t_tensor = torch.tensor(t_norm,  dtype=torch.float32).view(-1, 1)
xy_full  = torch.tensor(np.column_stack([x_ref, y_ref]), dtype=torch.float32)

# Sparse training data -- same points used by ALL three methods
train_mask = t_dense <= TRAIN_FRAC * T_SIM
train_idx  = np.where(train_mask)[0][::15]          # every 15th point
t_data     = t_tensor[train_idx]                     # (N_tr, 1)  normalised time
xy_data    = xy_full[train_idx]                      # (N_tr, 2)  physical position

# Phase-space data for HNN: z=(x,y,vx,vy), dz/dt=(vx,vy,ax,ay)
z_train  = torch.tensor(
    np.column_stack([x_ref[train_idx], y_ref[train_idx],
                     vx_ref[train_idx], vy_ref[train_idx]]),
    dtype=torch.float32)
dz_train = torch.tensor(
    np.column_stack([vx_ref[train_idx], vy_ref[train_idx],
                     ax_ref[train_idx], ay_ref[train_idx]]),
    dtype=torch.float32)

SPLIT = int(TRAIN_FRAC * N_DENSE)
print(f"\n  Training points : {len(train_idx)}  (from {N_DENSE} total)")
print(f"  Extrapolation   : t in [{t_dense[SPLIT]:.2f}, {T_SIM:.2f}]  "
      f"({N_DENSE - SPLIT} points)")

# =============================================================================
# 4.  ARCHITECTURES
# =============================================================================
class FCN(nn.Module):
    """Fully-connected: normalised_time -> (x, y)."""
    def __init__(self, n_hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


class HamiltonianNN(nn.Module):
    """
    Hamiltonian Neural Network (Greydanus, Dzamba, Yosinski -- NeurIPS 2019).

    Learns a scalar H : R^4 -> R from phase-space training data.
    Hamilton's equations give the dynamics:
        dq/dt =  dH/dp   =>  dx/dt = dH/dpx,   dy/dt = dH/dpy
        dp/dt = -dH/dq   =>  dpx/dt= -dH/dx,   dpy/dt= -dH/dy
    The LEARNED H is exactly conserved by construction (antisymmetry of J).
    """
    def __init__(self, n_hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(4, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

    def time_derivative(self, z):
        """
        Compute dz/dt = J * grad H  via autograd.
        z shape: (N, 4)   -- columns: [x, y, px, py]
        returns : (N, 4)  -- columns: [dx/dt, dy/dt, dpx/dt, dpy/dt]
        """
        z_r = z.detach().requires_grad_(True)
        H   = self.net(z_r).sum()
        dH  = torch.autograd.grad(H, z_r, create_graph=True)[0]  # (N,4)
        # [dH/dx, dH/dy, dH/dpx, dH/dpy]
        dqdt = dH[:, 2:4]           # dq/dt =  dH/dp
        dpdt = -dH[:, 0:2]          # dp/dt = -dH/dq
        return torch.cat([dqdt, dpdt], dim=1)

# =============================================================================
# 5.  HELPERS
# =============================================================================
def save_gif(outfile, files, fps=10):
    imgs = [Image.open(f) for f in files]
    imgs[0].save(fp=outfile, format="GIF", append_images=imgs[1:],
                 save_all=True, duration=int(1000/fps), loop=0)


def plot_frame(t_np, xy_ref_np, xy_pred_np,
               t_data_np, xy_data_np, epoch, label, color, filename):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(xy_ref_np[:,0], xy_ref_np[:,1], "#2ecc71", lw=2,
            label="Exact", zorder=3)
    a1.plot(xy_pred_np[:,0], xy_pred_np[:,1], color=color, lw=2.5,
            ls="--", alpha=0.85, label=f"{label} pred.", zorder=4)
    a1.scatter(xy_data_np[:,0], xy_data_np[:,1], s=60, c="#f39c12",
               zorder=5, edgecolors="k", lw=0.5, label="Train data")
    a1.scatter([0], [0], s=200, c="gold", marker="*", zorder=6,
               edgecolors="orange", lw=1, label="Star")
    a1.set_aspect("equal")
    a1.set_title(f"Orbit -- epoch {epoch:,}", fontweight="bold")
    a1.set_xlabel("x"); a1.set_ylabel("y"); a1.legend(fontsize=9)

    a2.plot(t_np, xy_ref_np[:,0], "#2ecc71", lw=1.8, label="x ref")
    a2.plot(t_np, xy_ref_np[:,1], "#27ae60", lw=1.8, ls="--", label="y ref")
    a2.plot(t_np, xy_pred_np[:,0], color=color, lw=2.5, label=f"x {label}")
    a2.plot(t_np, xy_pred_np[:,1], color=color, lw=2.5, ls="--",
            label=f"y {label}")
    a2.axvspan(0, TRAIN_FRAC, alpha=0.08, color="orange", label="Train")
    a2.set_xlabel("t (normalised)"); a2.set_ylabel("Position")
    a2.set_title("Position vs Time", fontweight="bold")
    a2.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=90, bbox_inches="tight", facecolor="white")
    plt.close()


def conservation(x, y, vx, vy):
    r = np.sqrt(x**2 + y**2) + 1e-12
    return (0.5*(vx**2+vy**2) - GM/r,   # energy
            x*vy - y*vx)                 # angular momentum


# =============================================================================
# 6.  STANDARD NN  (data-only baseline)
# =============================================================================
print("\n" + "-"*65)
print("  [1/3]  Standard NN  (data only)")
print("-"*65)

torch.manual_seed(42)
nn_model  = FCN()
nn_optim  = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
NN_EPOCHS = 5000
nn_losses = []
files_nn  = []

for i in range(NN_EPOCHS):
    nn_optim.zero_grad()
    loss = torch.mean((nn_model(t_data) - xy_data)**2)
    loss.backward(); nn_optim.step()
    nn_losses.append(loss.item())

    if (i+1) % 250 == 0:
        with torch.no_grad():
            xy_pred = nn_model(t_tensor).numpy()
        fn = f"plots/nn_{i+1:07d}.png"
        plot_frame(t_norm, xy_full.numpy(), xy_pred,
                   t_data.numpy(), xy_data.numpy(),
                   i+1, "NN", "#3498db", fn)
        files_nn.append(fn)
        if (i+1) % 2500 == 0:
            print(f"    epoch {i+1:5d}  loss={loss.item():.3e}")

save_gif("nn_kepler.gif", files_nn, fps=8)
print("  nn_kepler.gif saved")

# =============================================================================
# 7.  IMPROVED PINN  (ODE residual + IC position & velocity enforcement)
# =============================================================================
print("\n" + "-"*65)
print("  [2/3]  Improved PINN  (5 key fixes applied)")
print("-"*65)
print("    N_COLLOC  : 500    (was 50)")
print("    LAMBDA_PHYS: 1.0   (was 1e-3)")
print("    LAMBDA_IC  : 10.0  (IC pos + vel, was absent)")
print("    time normalised to [0,1]")
print("    StepLR decay  (halved every 10 k epochs)")

N_COLLOC     = 500
LAMBDA_PHYS  = 1.0
LAMBDA_IC    = 10.0
PINN_EPOCHS  = 30_000
LR_PINN      = 5e-4
T_SIM_t      = torch.tensor(T_SIM, dtype=torch.float32)

# Collocation points: full normalised domain [0, 1]
t_coll = torch.linspace(0.0, 1.0, N_COLLOC).view(-1, 1).requires_grad_(True)

# IC tensors (tau = 0 => t = 0)
t_ic0        = torch.zeros(1, 1)
ic_pos_tgt   = torch.tensor([[x0, y0]], dtype=torch.float32)
# velocities in normalised time:  dx/dtau = T_SIM * dx/dt = T_SIM * vx
ic_vel_tgt   = torch.tensor([[vx0 * T_SIM, vy0 * T_SIM]], dtype=torch.float32)

torch.manual_seed(42)
pinn_model  = FCN()
pinn_optim  = torch.optim.Adam(pinn_model.parameters(), lr=LR_PINN)
pinn_sched  = torch.optim.lr_scheduler.StepLR(
                  pinn_optim, step_size=10_000, gamma=0.5)

pinn_loss_total = []; pinn_loss_data = []
pinn_loss_phys  = []; pinn_loss_ic   = []
files_pinn = []

for i in range(PINN_EPOCHS):
    pinn_optim.zero_grad()

    # --- data loss -----------------------------------------------------------
    loss_d = torch.mean((pinn_model(t_data) - xy_data)**2)

    # --- IC position loss ----------------------------------------------------
    loss_ic_pos = torch.mean((pinn_model(t_ic0) - ic_pos_tgt)**2)

    # --- IC velocity loss (autograd at tau=0) --------------------------------
    t_ic_r   = t_ic0.clone().requires_grad_(True)
    pred_ic  = pinn_model(t_ic_r)
    xic = pred_ic[:, 0:1]; yic = pred_ic[:, 1:2]
    vxic = torch.autograd.grad(xic, t_ic_r, torch.ones_like(xic),
                                create_graph=True)[0]
    vyic = torch.autograd.grad(yic, t_ic_r, torch.ones_like(yic),
                                create_graph=True)[0]
    loss_ic_vel = ((vxic - ic_vel_tgt[:, 0:1])**2 +
                   (vyic - ic_vel_tgt[:, 1:2])**2).mean()
    loss_ic = loss_ic_pos + loss_ic_vel

    # --- physics (ODE residual) loss -----------------------------------------
    pred_p = pinn_model(t_coll)
    xp = pred_p[:, 0:1]; yp = pred_p[:, 1:2]

    vx_p = torch.autograd.grad(xp,  t_coll, torch.ones_like(xp),
                                create_graph=True)[0]
    vy_p = torch.autograd.grad(yp,  t_coll, torch.ones_like(yp),
                                create_graph=True)[0]
    ax_p = torch.autograd.grad(vx_p, t_coll, torch.ones_like(vx_p),
                                create_graph=True)[0]
    ay_p = torch.autograd.grad(vy_p, t_coll, torch.ones_like(vy_p),
                                create_graph=True)[0]

    # Chain rule: d^2x/dtau^2 = T_SIM^2 * d^2x/dt^2 = T_SIM^2 * (-GM*x/r^3)
    r_p   = torch.sqrt(xp**2 + yp**2 + 1e-8)
    res_x = ax_p - T_SIM_t**2 * (-GM * xp / r_p**3)
    res_y = ay_p - T_SIM_t**2 * (-GM * yp / r_p**3)
    loss_p = torch.mean(res_x**2 + res_y**2)

    # --- total ---------------------------------------------------------------
    loss = loss_d + LAMBDA_PHYS * loss_p + LAMBDA_IC * loss_ic
    loss.backward()
    pinn_optim.step()
    pinn_sched.step()

    pinn_loss_total.append(loss.item())
    pinn_loss_data.append(loss_d.item())
    pinn_loss_phys.append(loss_p.item())
    pinn_loss_ic.append(loss_ic.item())

    if (i+1) % 1000 == 0:
        with torch.no_grad():
            xy_pred = pinn_model(t_tensor).numpy()
        fn = f"plots/pinn_{i+1:07d}.png"
        plot_frame(t_norm, xy_full.numpy(), xy_pred,
                   t_data.numpy(), xy_data.numpy(),
                   i+1, "PINN", "#e74c3c", fn)
        files_pinn.append(fn)
        if (i+1) % 5000 == 0:
            print(f"    epoch {i+1:6d} | total={loss.item():.3e} "
                  f"data={loss_d.item():.3e} "
                  f"phys={loss_p.item():.3e} "
                  f"ic={loss_ic.item():.3e}")

save_gif("pinn_kepler.gif", files_pinn, fps=8)
print("  pinn_kepler.gif saved")

# =============================================================================
# 8.  HAMILTONIAN NN  (Greydanus et al. NeurIPS 2019)
# =============================================================================
print("\n" + "-"*65)
print("  [3/3]  Hamiltonian Neural Network (HNN)")
print("-"*65)
print("  Input : z = (x, y, px, py)")
print("  Output: scalar H(z)  [learned Hamiltonian]")
print("  Loss  : MSE on Hamilton's equations at training data points")
print("  Infer : scipy.solve_ivp with dz/dt = J * grad_H")

torch.manual_seed(42)
hnn_model  = HamiltonianNN(n_hidden=128, n_layers=3)
hnn_optim  = torch.optim.Adam(hnn_model.parameters(), lr=1e-3)
hnn_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                 hnn_optim, T_max=10_000, eta_min=1e-5)
HNN_EPOCHS = 10_000
hnn_losses = []

for i in range(HNN_EPOCHS):
    hnn_optim.zero_grad()
    dz_pred = hnn_model.time_derivative(z_train)
    loss    = torch.mean((dz_pred - dz_train)**2)
    loss.backward()
    hnn_optim.step()
    hnn_sched.step()
    hnn_losses.append(loss.item())
    if (i+1) % 2000 == 0:
        print(f"    epoch {i+1:6d}  loss={loss.item():.3e}")

print("  HNN trained")

# --- integrate with scipy.solve_ivp ------------------------------------------
print("  Integrating HNN trajectory (RK45)...", end="", flush=True)

def hnn_rhs(t, state):
    with torch.enable_grad():
        z  = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        z  = z.requires_grad_(True)
        H  = hnn_model(z).sum()
        dH = torch.autograd.grad(H, z)[0].squeeze()
    return [dH[2].item(), dH[3].item(), -dH[0].item(), -dH[1].item()]

sol_hnn = solve_ivp(
    hnn_rhs,
    t_span=[0.0, T_SIM],
    y0=[x0, y0, vx0, vy0],
    t_eval=t_dense,
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)

if sol_hnn.success and sol_hnn.y.shape[1] == N_DENSE:
    x_hnn, y_hnn, vx_hnn, vy_hnn = sol_hnn.y
    print(f" done ({sol_hnn.nfev} func evals)")
else:
    print(f"\n  WARNING: {sol_hnn.message}")
    n = sol_hnn.y.shape[1]
    pad = lambda a: np.pad(a, (0, N_DENSE-n), constant_values=np.nan)
    x_hnn, y_hnn = pad(sol_hnn.y[0]), pad(sol_hnn.y[1])
    vx_hnn, vy_hnn = pad(sol_hnn.y[2]), pad(sol_hnn.y[3])

print("  HNN trajectory integrated")

# =============================================================================
# 9.  CONSERVATION LAWS
# =============================================================================
with torch.no_grad():
    xy_nn   = nn_model(t_tensor).numpy()
    xy_pinn = pinn_model(t_tensor).numpy()

dt = t_dense[1] - t_dense[0]
# Velocities for NN/PINN estimated via central finite differences
vx_nn_fd   = np.gradient(xy_nn[:,0],   dt)
vy_nn_fd   = np.gradient(xy_nn[:,1],   dt)
vx_pinn_fd = np.gradient(xy_pinn[:,0], dt)
vy_pinn_fd = np.gradient(xy_pinn[:,1], dt)

E_nn,   L_nn   = conservation(xy_nn[:,0],   xy_nn[:,1],   vx_nn_fd,   vy_nn_fd)
E_pinn, L_pinn = conservation(xy_pinn[:,0], xy_pinn[:,1], vx_pinn_fd, vy_pinn_fd)
E_hnn,  L_hnn  = conservation(x_hnn, y_hnn, vx_hnn, vy_hnn)

# Learned Hamiltonian along HNN trajectory
z_hnn_traj = torch.tensor(
    np.column_stack([x_hnn, y_hnn, vx_hnn, vy_hnn]),
    dtype=torch.float32)
with torch.no_grad():
    H_learned = hnn_model(z_hnn_traj).numpy().flatten()

# =============================================================================
# 10.  ERROR STATISTICS
# =============================================================================
err_nn   = np.sqrt((xy_nn[:,0]  -x_ref)**2 + (xy_nn[:,1]  -y_ref)**2)
err_pinn = np.sqrt((xy_pinn[:,0]-x_ref)**2 + (xy_pinn[:,1]-y_ref)**2)
err_hnn  = np.sqrt((x_hnn      -x_ref)**2 + (y_hnn       -y_ref)**2)

# =============================================================================
# 11.  COMPREHENSIVE COMPARISON FIGURE  (8 panels)
# =============================================================================
COLORS = dict(Exact="#2ecc71", NN="#3498db", PINN="#e74c3c", HNN="#9b59b6")

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
ax_orb.plot(x_ref, y_ref, COLORS["Exact"], lw=3, label="Exact", zorder=3)
for lbl, xarr, yarr in [("NN",   xy_nn[:,0],   xy_nn[:,1]),
                         ("PINN", xy_pinn[:,0], xy_pinn[:,1]),
                         ("HNN",  x_hnn,        y_hnn)]:
    ax_orb.plot(xarr, yarr, COLORS[lbl], lw=2,
                ls="-." if lbl=="NN" else ("--" if lbl=="PINN" else ":"),
                alpha=0.9, label=lbl)
ax_orb.scatter(xy_data.numpy()[:,0], xy_data.numpy()[:,1],
               s=70, c="#f39c12", zorder=5, edgecolors="k", lw=0.5,
               label="Training data")
ax_orb.scatter([0],[0], s=300, c="gold", marker="*", zorder=6,
               edgecolors="darkorange", lw=1.5, label="Star (focus)")
ax_orb.set_aspect("equal")
ax_orb.set_title("(A) Orbital Trajectories", fontweight="bold", fontsize=12)
ax_orb.set_xlabel("x"); ax_orb.set_ylabel("y")
ax_orb.legend(fontsize=9, ncol=2, framealpha=0.9)

# (B) x(t)
ax_xt.plot(t_dense, x_ref, COLORS["Exact"], lw=2, label="Exact")
for lbl, arr in [("NN", xy_nn[:,0]), ("PINN", xy_pinn[:,0]), ("HNN", x_hnn)]:
    ax_xt.plot(t_dense, arr, COLORS[lbl], lw=1.5,
               ls="-." if lbl=="NN" else ("--" if lbl=="PINN" else ":"),
               label=lbl)
ax_xt.axvspan(0, TRAIN_FRAC*T_SIM, alpha=0.08, color="orange")
ax_xt.set_title("(B) x(t)", fontweight="bold", fontsize=12)
ax_xt.set_xlabel("Time (s)"); ax_xt.set_ylabel("x")
ax_xt.legend(fontsize=8); ax_xt.grid(alpha=0.3)

# (C) Position error (log)
for lbl, err in [("NN", err_nn), ("PINN", err_pinn), ("HNN", err_hnn)]:
    ax_err.semilogy(t_dense, err + 1e-10, COLORS[lbl], lw=2, label=lbl)
ax_err.axvline(TRAIN_FRAC*T_SIM, color="gray", ls=":", lw=1.5,
               label="Train end")
ax_err.set_title("(C) Position Error |Dr|", fontweight="bold", fontsize=12)
ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("|Dr| (log)")
ax_err.legend(fontsize=9); ax_err.grid(alpha=0.3)

# (D) Energy conservation
ax_E.plot(t_dense, E_ref, COLORS["Exact"], lw=2.5, label="Exact E")
for lbl, E in [("NN", E_nn), ("PINN", E_pinn), ("HNN", E_hnn)]:
    ax_E.plot(t_dense, E, COLORS[lbl], lw=1.5, ls="--", alpha=0.85,
              label=lbl)
ax_E.set_title("(D) Energy E(t)", fontweight="bold", fontsize=12)
ax_E.set_xlabel("Time"); ax_E.set_ylabel("Energy")
ax_E.legend(fontsize=8); ax_E.grid(alpha=0.3)

# (E) Angular momentum conservation
ax_L.plot(t_dense, L_ref, COLORS["Exact"], lw=2.5, label="Exact L")
for lbl, L in [("NN", L_nn), ("PINN", L_pinn), ("HNN", L_hnn)]:
    ax_L.plot(t_dense, L, COLORS[lbl], lw=1.5, ls="--", alpha=0.85,
              label=lbl)
ax_L.set_title("(E) Angular Momentum L(t)", fontweight="bold", fontsize=12)
ax_L.set_xlabel("Time"); ax_L.set_ylabel("L")
ax_L.legend(fontsize=8); ax_L.grid(alpha=0.3)

# (F) Loss curves
ax_loss.semilogy(nn_losses, COLORS["NN"], lw=1.5, label="NN data loss")
ax_loss.semilogy(pinn_loss_data, COLORS["PINN"], lw=1.5,
                 label="PINN data loss")
ax_loss.semilogy([LAMBDA_PHYS*v for v in pinn_loss_phys], "#e74c3c",
                 lw=1, ls=":", alpha=0.6, label="lam*PINN phys")
ax_loss.semilogy(hnn_losses, COLORS["HNN"], lw=1.5, label="HNN loss")
ax_loss.set_title("(F) Training Loss Curves", fontweight="bold", fontsize=12)
ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss (log)")
ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)

# (G) Learned H vs true energy
ax_Hl.plot(t_dense, E_ref, COLORS["Exact"], lw=2.5, label="True energy E")
ax_Hl.plot(t_dense, H_learned, COLORS["HNN"], lw=2, ls="--",
           label="Learned H (HNN)")
ax_Hl.set_title("(G) HNN Learned H vs True Energy", fontweight="bold",
                fontsize=12)
ax_Hl.set_xlabel("Time"); ax_Hl.set_ylabel("Value")
ax_Hl.legend(fontsize=9); ax_Hl.grid(alpha=0.3)

# (H) HNN orbit close-up
ax_Hnn.plot(x_ref, y_ref, COLORS["Exact"], lw=3, label="Exact")
ax_Hnn.plot(x_hnn, y_hnn, COLORS["HNN"], lw=2.5, ls="--",
            alpha=0.9, label="HNN")
ax_Hnn.scatter([0],[0], s=200, c="gold", marker="*", zorder=6,
               edgecolors="darkorange", lw=1.5)
ax_Hnn.set_aspect("equal")
ax_Hnn.set_title("(H) HNN Orbit Detail", fontweight="bold", fontsize=12)
ax_Hnn.set_xlabel("x"); ax_Hnn.set_ylabel("y")
ax_Hnn.legend(fontsize=9)

fig.suptitle("Kepler Orbit: NN vs Improved PINN vs HNN  |  e=0.5, GM=1",
             fontsize=15, fontweight="bold")
plt.savefig("plots/kepler_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("\n  plots/kepler_comparison.png saved")

# =============================================================================
# 12.  SUMMARY TABLE
# =============================================================================
print("\n" + "="*75)
print(f"  {'Method':<18} {'Train MAE':>12} {'Extrapol MAE':>14} "
      f"{'|dE| extrapol':>15} {'|dL| extrapol':>15}")
print("="*75)
for name, err, E_arr, L_arr in [
        ("Standard NN",   err_nn,   E_nn,   L_nn),
        ("Improved PINN", err_pinn, E_pinn, L_pinn),
        ("HNN",           err_hnn,  E_hnn,  L_hnn)]:
    tr  = np.nanmean(err[:SPLIT])
    ex  = np.nanmean(err[SPLIT:])
    dE  = np.nanmean(np.abs(E_arr[SPLIT:] - E_ref[SPLIT:]))
    dL  = np.nanmean(np.abs(L_arr[SPLIT:] - L_ref[SPLIT:]))
    print(f"  {name:<18} {tr:>12.5f} {ex:>14.5f} {dE:>15.6f} {dL:>15.6f}")

print("="*75)
print(f"\n  Exact: E = {E_ref[0]:.6f} (constant, analytical = -GM/2a = -0.5)")
print(f"  Exact: L = {L_ref[0]:.6f} (constant, sqrt(GM*a*(1-e^2)) = sqrt(0.75))")
print("\n  HNN note: H_learned is EXACTLY conserved along the HNN trajectory.")
print("  PINN note: conservation emerges if ODE residual is well minimised.")
print("  NN  note: no conservation mechanism; large energy drift expected.")
print("="*75)
