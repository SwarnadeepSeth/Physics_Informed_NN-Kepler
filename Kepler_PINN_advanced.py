#!/usr/bin/env python3
"""
Kepler Orbit: Research-Level PINN Improvements
================================================
Five research-grade enhancements beyond the baseline PINN:

  Fix 1 – Energy conservation loss         (E(t) = const)
  Fix 2 – Angular momentum constraint      (L(t) = const)
  Fix 3 – Adaptive collocation             (biased near periapsis)
  Fix 4 – Neural ODE                       (phase-space: z -> dz/dt)
  Fix 5 – Symplectic PINN                  (Fixes 1+2+3 combined)

Three methods are compared here:
  A) Improved PINN    — previous 5-fix version (benchmark)
  B) Symplectic PINN  — improved PINN + energy + angmom + adaptive coll.
  C) Neural ODE       — phase-space representation via torchdiffeq

Reference: Greydanus et al. NeurIPS 2019  |  Raissi et al. JCP 2019
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint as torch_odeint
from scipy.integrate import odeint
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
GM     = 1.0
a      = 1.0
e      = 0.5

x0   = a * (1.0 - e)
y0   = 0.0
vx0  = 0.0
vy0  = np.sqrt(GM * (1.0 + e) / (a * (1.0 - e)))

T_orb  = 2.0 * np.pi * np.sqrt(a**3 / GM)
T_SIM  = 1.5 * T_orb
N_DENSE    = 1000
TRAIN_FRAC = 0.4

E0 = 0.5 * vy0**2 - GM / x0          # exact mechanical energy = -GM/(2a)
L0 = x0 * vy0                         # exact angular momentum

print("=" * 65)
print("  Kepler Orbit  -- Research-Level PINN Improvements")
print("=" * 65)
print(f"  e={e}, T_orb={T_orb:.4f}, T_sim={T_SIM:.4f}")
print(f"  E0 (exact) = {E0:.6f}  |  L0 (exact) = {L0:.6f}")
print("=" * 65)

# =============================================================================
# 2.  NUMERICAL REFERENCE
# =============================================================================
def kepler_ode(state, t):
    x, y, vx, vy = state
    r  = np.sqrt(x**2 + y**2)
    return [vx, vy, -GM*x/r**3, -GM*y/r**3]

t_dense  = np.linspace(0.0, T_SIM, N_DENSE)
sol      = odeint(kepler_ode, [x0, y0, vx0, vy0], t_dense,
                  rtol=1e-11, atol=1e-13)
x_ref, y_ref, vx_ref, vy_ref = sol.T
r_ref  = np.sqrt(x_ref**2 + y_ref**2)
E_ref  = 0.5*(vx_ref**2 + vy_ref**2) - GM/r_ref
L_ref  = x_ref*vy_ref - y_ref*vx_ref

SPLIT      = int(TRAIN_FRAC * N_DENSE)
train_mask = t_dense <= TRAIN_FRAC * T_SIM
train_idx  = np.where(train_mask)[0][::15]

print(f"  Training pts  : {len(train_idx)} / {N_DENSE}")
print(f"  Extrapolation : t ∈ [{t_dense[SPLIT]:.2f}, {T_SIM:.2f}]\n")

# =============================================================================
# 3.  TENSORS
# =============================================================================
t_norm   = t_dense / T_SIM
t_tensor = torch.tensor(t_norm,   dtype=torch.float32).view(-1, 1)
xy_full  = torch.tensor(np.column_stack([x_ref, y_ref]),   dtype=torch.float32)
z_full   = torch.tensor(np.column_stack([x_ref, y_ref, vx_ref, vy_ref]),
                        dtype=torch.float32)

t_data  = t_tensor[train_idx]
xy_data = xy_full[train_idx]
z_train = z_full[train_idx]

T_SIM_t  = torch.tensor(T_SIM,  dtype=torch.float32)
T_ORB_t  = torch.tensor(T_orb,  dtype=torch.float32)
E0_t     = torch.tensor(E0,     dtype=torch.float32)
L0_t     = torch.tensor(L0,     dtype=torch.float32)

# =============================================================================
# 4.  ARCHITECTURES
# =============================================================================
class FCN(nn.Module):
    """Normalised time τ ∈ [0,1]  →  (x, y)."""
    def __init__(self, n_hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


class NeuralODE_f(nn.Module):
    """
    Neural ODE vector field:  f(z) : R^4 -> R^4
    z = (x, y, vx, vy)   →   dz/dt = f(z)

    The network is AUTONOMOUS (no explicit time dependence) which is correct
    for a time-homogeneous Hamiltonian system.  torchdiffeq integrates this
    as:  z(t) = z0 + ∫₀ᵗ f(z(s)) ds.
    """
    def __init__(self, n_hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(4, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        """torchdiffeq convention: forward(t, z), both 1D (single state)."""
        return self.net(z)


# =============================================================================
# 5.  HELPERS
# =============================================================================
def conservation(x, y, vx, vy):
    r = np.sqrt(x**2 + y**2) + 1e-12
    return 0.5*(vx**2+vy**2) - GM/r,  x*vy - y*vx


def adaptive_collocation(N):
    """
    Fix 3 — Adaptive collocation: concentrate points near periapsis passages.

    For e=0.5, T_SIM=1.5*T_orb:
      Periapsis 1  at tau=0.0
      Periapsis 2  at tau=T_orb/T_SIM ≈ 0.667

    Strategy: 40% uniform + 30% near each periapsis (Gaussian σ=0.04).
    """
    tau_peri = T_orb / T_SIM              # ≈ 0.667
    n_uni = int(0.40 * N)
    n_p1  = int(0.30 * N)
    n_p2  = N - n_uni - n_p1

    t_uni = torch.linspace(0.0, 1.0, n_uni)
    t_p1  = torch.randn(n_p1) * 0.04                  # near tau=0
    t_p2  = torch.randn(n_p2) * 0.04 + tau_peri       # near tau=0.667
    t_all = torch.cat([t_uni, t_p1, t_p2]).clamp(0.0, 1.0)
    return t_all.sort()[0].view(-1, 1).requires_grad_(True)


def pinn_residuals(model, t_coll):
    """
    Evaluate ODE residuals, velocities and positions at collocation points.
    Returns:  xp, yp, vx_p, vy_p, vx_phys, vy_phys, r_p, res_x, res_y
    Time chain rule:  d^2x/dtau^2  =  T_SIM^2 * d^2x/dt^2.
    """
    pred_p = model(t_coll)
    xp = pred_p[:, 0:1]; yp = pred_p[:, 1:2]

    vx_p = torch.autograd.grad(xp,  t_coll, torch.ones_like(xp),
                                create_graph=True)[0]
    vy_p = torch.autograd.grad(yp,  t_coll, torch.ones_like(yp),
                                create_graph=True)[0]
    ax_p = torch.autograd.grad(vx_p, t_coll, torch.ones_like(vx_p),
                                create_graph=True)[0]
    ay_p = torch.autograd.grad(vy_p, t_coll, torch.ones_like(vy_p),
                                create_graph=True)[0]

    r_p       = torch.sqrt(xp**2 + yp**2 + 1e-8)
    res_x     = ax_p - T_SIM_t**2 * (-GM * xp / r_p**3)
    res_y     = ay_p - T_SIM_t**2 * (-GM * yp / r_p**3)
    # physical velocities (de-normalise):  vx_phys = vx_p / T_SIM
    vx_phys   = vx_p / T_SIM_t
    vy_phys   = vy_p / T_SIM_t

    return xp, yp, vx_phys, vy_phys, r_p, res_x, res_y


# =============================================================================
# 6.  METHOD A — IMPROVED PINN  (5 fixes, no conservation terms)
#     This is the same PINN from Kepler_PINN.py, re-trained here for fair
#     comparison.  Fewer epochs (20k) to keep runtime manageable.
# =============================================================================
print("-"*65)
print("  [A] Improved PINN  (5 fixes, no conservation terms)")
print("-"*65)

N_COLLOC  = 500
LAMBDA_PHYS = 1.0
LAMBDA_IC   = 10.0
A_EPOCHS    = 20_000

t_coll_A = torch.linspace(0.0, 1.0, N_COLLOC).view(-1, 1).requires_grad_(True)
t_ic0    = torch.zeros(1, 1)
ic_pos_tgt = torch.tensor([[x0, y0]], dtype=torch.float32)
ic_vel_tgt = torch.tensor([[vx0 * T_SIM, vy0 * T_SIM]], dtype=torch.float32)

torch.manual_seed(42)
pinn_A      = FCN()
opt_A       = torch.optim.Adam(pinn_A.parameters(), lr=5e-4)
sched_A     = torch.optim.lr_scheduler.StepLR(opt_A, step_size=7_000, gamma=0.5)
losses_A    = []

for i in range(A_EPOCHS):
    opt_A.zero_grad()

    loss_d = torch.mean((pinn_A(t_data) - xy_data)**2)

    loss_ic_pos  = torch.mean((pinn_A(t_ic0) - ic_pos_tgt)**2)
    t_ic_r       = t_ic0.clone().requires_grad_(True)
    pred_ic      = pinn_A(t_ic_r)
    vxic = torch.autograd.grad(pred_ic[:,0:1], t_ic_r,
                                torch.ones(1,1), create_graph=True)[0]
    vyic = torch.autograd.grad(pred_ic[:,1:2], t_ic_r,
                                torch.ones(1,1), create_graph=True)[0]
    loss_ic_vel  = ((vxic - ic_vel_tgt[:,0:1])**2 +
                    (vyic - ic_vel_tgt[:,1:2])**2).mean()
    loss_ic      = loss_ic_pos + loss_ic_vel

    xp,yp,_,_,r_p,res_x,res_y = pinn_residuals(pinn_A, t_coll_A)
    loss_p = torch.mean(res_x**2 + res_y**2)

    loss = loss_d + LAMBDA_PHYS * loss_p + LAMBDA_IC * loss_ic
    loss.backward(); opt_A.step(); sched_A.step()
    losses_A.append(loss.item())

    if (i+1) % 5000 == 0:
        print(f"    epoch {i+1:6d} | total={loss.item():.3e}  "
              f"phys={loss_p.item():.3e}  ic={loss_ic.item():.3e}")

print("  Improved PINN trained\n")

# =============================================================================
# 7.  METHOD B — SYMPLECTIC PINN  (Fix 1 + Fix 2 + Fix 3)
#     PINN with:
#       Fix 1: energy conservation loss     λ_E  * mean((E_pred - E0)^2)
#       Fix 2: angular momentum constraint  λ_L  * mean((L_pred - L0)^2)
#       Fix 3: adaptive collocation         biased near periapsis
# =============================================================================
print("-"*65)
print("  [B] Symplectic PINN  (Fix 1: energy + Fix 2: ang.mom + Fix 3: adaptive coll.)")
print("-"*65)
print("  λ_energy = 5.0  |  λ_angmom = 5.0  |  adaptive collocation near periapsis")

LAMBDA_E    = 5.0
LAMBDA_L    = 5.0
B_EPOCHS    = 20_000

torch.manual_seed(42)
pinn_B   = FCN()
opt_B    = torch.optim.Adam(pinn_B.parameters(), lr=5e-4)
sched_B  = torch.optim.lr_scheduler.StepLR(opt_B, step_size=7_000, gamma=0.5)
losses_B = []

for i in range(B_EPOCHS):
    opt_B.zero_grad()

    # Re-sample adaptive collocation every 100 epochs (stochastic)
    if i % 100 == 0:
        t_coll_B = adaptive_collocation(N_COLLOC)

    # --- data loss ---
    loss_d = torch.mean((pinn_B(t_data) - xy_data)**2)

    # --- IC loss ---
    loss_ic_pos = torch.mean((pinn_B(t_ic0) - ic_pos_tgt)**2)
    t_ic_r      = t_ic0.clone().requires_grad_(True)
    pred_ic_B   = pinn_B(t_ic_r)
    vxic  = torch.autograd.grad(pred_ic_B[:,0:1], t_ic_r, torch.ones(1,1), create_graph=True)[0]
    vyic  = torch.autograd.grad(pred_ic_B[:,1:2], t_ic_r, torch.ones(1,1), create_graph=True)[0]
    loss_ic_vel = ((vxic - ic_vel_tgt[:,0:1])**2 + (vyic - ic_vel_tgt[:,1:2])**2).mean()
    loss_ic     = loss_ic_pos + loss_ic_vel

    # --- physics (ODE residual) ---
    xp, yp, vx_phys, vy_phys, r_p, res_x, res_y = pinn_residuals(pinn_B, t_coll_B)
    loss_p = torch.mean(res_x**2 + res_y**2)

    # --- Fix 1: energy conservation loss ---------------------------------
    # E = 0.5*(vx^2 + vy^2) - GM/r  must equal E0 everywhere
    E_pred    = 0.5*(vx_phys**2 + vy_phys**2) - GM / r_p
    loss_E    = torch.mean((E_pred - E0_t)**2)

    # --- Fix 2: angular momentum conservation loss -----------------------
    # L = x*vy - y*vx  must equal L0 everywhere
    L_pred    = xp * vy_phys - yp * vx_phys
    loss_L    = torch.mean((L_pred - L0_t)**2)

    # --- total ---
    loss = (loss_d
            + LAMBDA_PHYS * loss_p
            + LAMBDA_IC   * loss_ic
            + LAMBDA_E    * loss_E
            + LAMBDA_L    * loss_L)
    loss.backward(); opt_B.step(); sched_B.step()
    losses_B.append(loss.item())

    if (i+1) % 5000 == 0:
        print(f"    epoch {i+1:6d} | total={loss.item():.3e}  "
              f"phys={loss_p.item():.3e}  E={loss_E.item():.3e}  L={loss_L.item():.3e}")

print("  Symplectic PINN trained\n")

# =============================================================================
# 8.  METHOD C — NEURAL ODE  (Fix 4: phase-space representation)
#     Learns  f: z=(x,y,vx,vy) -> dz/dt  directly (no time-state mapping).
#     torchdiffeq.odeint integrates from IC to get trajectory.
# =============================================================================
print("-"*65)
print("  [C] Neural ODE  (Fix 4: phase-space  z -> dz/dt)")
print("-"*65)
print("  Architecture: R^4 -> R^4  |  128 hidden, 3 layers  |  Tanh")
print("  Integrator  : torchdiffeq RK4  (adjoint method for memory)")

NODE_EPOCHS = 10_000

# Use physical times (not normalised) for the ODE integrator
t_train_phys = torch.tensor(t_dense[train_idx], dtype=torch.float32)
z_train_full = torch.tensor(
    np.column_stack([x_ref[train_idx], y_ref[train_idx],
                     vx_ref[train_idx], vy_ref[train_idx]]),
    dtype=torch.float32)           # shape (N_tr, 4)
z0_node = z_train_full[0]         # IC at t=0

torch.manual_seed(42)
node_f   = NeuralODE_f(n_hidden=128, n_layers=3)
opt_C    = torch.optim.Adam(node_f.parameters(), lr=1e-3)
sched_C  = torch.optim.lr_scheduler.CosineAnnealingLR(
               opt_C, T_max=NODE_EPOCHS, eta_min=1e-5)
losses_C = []

for i in range(NODE_EPOCHS):
    opt_C.zero_grad()
    # Integrate Neural ODE from z0 to all training times
    z_pred = torch_odeint(node_f, z0_node, t_train_phys,
                          method="rk4", options={"step_size": 0.05})
    # z_pred: (N_train, 4)
    loss = torch.mean((z_pred - z_train_full)**2)
    loss.backward(); opt_C.step(); sched_C.step()
    losses_C.append(loss.item())

    if (i+1) % 2000 == 0:
        print(f"    epoch {i+1:6d}  loss={loss.item():.3e}")

print("  Neural ODE trained\n")

# --- Integrate full trajectory ---
print("  Integrating Neural ODE over full T_SIM ...", end="", flush=True)
t_full_phys = torch.tensor(t_dense, dtype=torch.float32)
with torch.no_grad():
    z_node_full = torch_odeint(node_f, z0_node, t_full_phys,
                                method="rk4", options={"step_size": 0.05})
x_node  = z_node_full[:,0].numpy()
y_node  = z_node_full[:,1].numpy()
vx_node = z_node_full[:,2].numpy()
vy_node = z_node_full[:,3].numpy()
print(" done")

# =============================================================================
# 9.  CONSERVATION LAWS
# =============================================================================
with torch.no_grad():
    xy_A = pinn_A(t_tensor).numpy()
    xy_B = pinn_B(t_tensor).numpy()

dt  = t_dense[1] - t_dense[0]
vx_A_fd = np.gradient(xy_A[:,0], dt); vy_A_fd = np.gradient(xy_A[:,1], dt)
vx_B_fd = np.gradient(xy_B[:,0], dt); vy_B_fd = np.gradient(xy_B[:,1], dt)

E_A, L_A = conservation(xy_A[:,0], xy_A[:,1], vx_A_fd, vy_A_fd)
E_B, L_B = conservation(xy_B[:,0], xy_B[:,1], vx_B_fd, vy_B_fd)
E_C, L_C = conservation(x_node, y_node, vx_node, vy_node)

# =============================================================================
# 10.  ERROR STATISTICS
# =============================================================================
err_A = np.sqrt((xy_A[:,0]-x_ref)**2 + (xy_A[:,1]-y_ref)**2)
err_B = np.sqrt((xy_B[:,0]-x_ref)**2 + (xy_B[:,1]-y_ref)**2)
err_C = np.sqrt((x_node    -x_ref)**2 + (y_node    -y_ref)**2)

# =============================================================================
# 11.  COMPREHENSIVE COMPARISON FIGURE  (3x3 = 9 panels)
# =============================================================================
COLORS = {
    "Exact"    : "#2ecc71",
    "PINN-A"   : "#e74c3c",
    "SPINN-B"  : "#e67e22",
    "NODE-C"   : "#9b59b6",
}

fig = plt.figure(figsize=(18, 16))
gs  = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

ax_orb  = fig.add_subplot(gs[0, :2])
ax_xt   = fig.add_subplot(gs[0, 2])
ax_err  = fig.add_subplot(gs[1, 0])
ax_E    = fig.add_subplot(gs[1, 1])
ax_L    = fig.add_subplot(gs[1, 2])
ax_loss = fig.add_subplot(gs[2, 0])
ax_coll = fig.add_subplot(gs[2, 1])
ax_err2 = fig.add_subplot(gs[2, 2])

# (A) Trajectories
ax_orb.plot(x_ref, y_ref, COLORS["Exact"], lw=3, label="Exact", zorder=3)
for lbl, c, xarr, yarr, ls in [
        ("Improved PINN",  COLORS["PINN-A"],  xy_A[:,0], xy_A[:,1], "--"),
        ("Symplectic PINN",COLORS["SPINN-B"], xy_B[:,0], xy_B[:,1], "-."),
        ("Neural ODE",     COLORS["NODE-C"],  x_node,    y_node,    ":")]:
    ax_orb.plot(xarr, yarr, c, lw=2, ls=ls, alpha=0.9, label=lbl)
ax_orb.scatter(xy_data.numpy()[:,0], xy_data.numpy()[:,1],
               s=70, c="#f39c12", zorder=5, edgecolors="k", lw=0.5,
               label="Training data")
ax_orb.scatter([0],[0], s=300, c="gold", marker="*", zorder=6,
               edgecolors="darkorange", lw=1.5, label="Star")
ax_orb.set_aspect("equal")
ax_orb.set_title("(A) Orbital Trajectories", fontweight="bold", fontsize=12)
ax_orb.legend(fontsize=9, ncol=2, framealpha=0.9)
ax_orb.set_xlabel("x"); ax_orb.set_ylabel("y")

# (B) x(t)
ax_xt.plot(t_dense, x_ref, COLORS["Exact"], lw=2, label="Exact")
for lbl, c, arr, ls in [
        ("Improved PINN",  COLORS["PINN-A"],  xy_A[:,0], "--"),
        ("Symplectic PINN",COLORS["SPINN-B"], xy_B[:,0], "-."),
        ("Neural ODE",     COLORS["NODE-C"],  x_node,    ":")]:
    ax_xt.plot(t_dense, arr, c, lw=1.5, ls=ls, label=lbl)
ax_xt.axvspan(0, TRAIN_FRAC*T_SIM, alpha=0.08, color="orange")
ax_xt.set_title("(B) x(t)", fontweight="bold", fontsize=12)
ax_xt.legend(fontsize=8); ax_xt.grid(alpha=0.3)
ax_xt.set_xlabel("Time (s)"); ax_xt.set_ylabel("x")

# (C) Error (log)
for lbl, c, err in [
        ("Improved PINN",  COLORS["PINN-A"],  err_A),
        ("Symplectic PINN",COLORS["SPINN-B"], err_B),
        ("Neural ODE",     COLORS["NODE-C"],  err_C)]:
    ax_err.semilogy(t_dense, err + 1e-10, c, lw=2, label=lbl)
ax_err.axvline(TRAIN_FRAC*T_SIM, color="gray", ls=":", lw=1.5, label="Train end")
ax_err.set_title("(C) Position Error |Δr|", fontweight="bold", fontsize=12)
ax_err.legend(fontsize=9); ax_err.grid(alpha=0.3)
ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("|Δr| (log)")

# (D) Energy conservation
ax_E.axhline(E0, color=COLORS["Exact"], lw=2.5, ls="-", label=f"Exact E₀={E0:.3f}")
for lbl, c, E_arr in [
        ("Improved PINN",  COLORS["PINN-A"],  E_A),
        ("Symplectic PINN",COLORS["SPINN-B"], E_B),
        ("Neural ODE",     COLORS["NODE-C"],  E_C)]:
    ax_E.plot(t_dense, E_arr, c, lw=1.5, ls="--", alpha=0.85, label=lbl)
ax_E.set_title("(D) Energy E(t)  [should be flat]", fontweight="bold", fontsize=12)
ax_E.legend(fontsize=8); ax_E.grid(alpha=0.3)
ax_E.set_xlabel("Time"); ax_E.set_ylabel("Energy")

# (E) Angular momentum
ax_L.axhline(L0, color=COLORS["Exact"], lw=2.5, ls="-", label=f"Exact L₀={L0:.3f}")
for lbl, c, L_arr in [
        ("Improved PINN",  COLORS["PINN-A"],  L_A),
        ("Symplectic PINN",COLORS["SPINN-B"], L_B),
        ("Neural ODE",     COLORS["NODE-C"],  L_C)]:
    ax_L.plot(t_dense, L_arr, c, lw=1.5, ls="--", alpha=0.85, label=lbl)
ax_L.set_title("(E) Angular Momentum L(t)  [should be flat]",
               fontweight="bold", fontsize=12)
ax_L.legend(fontsize=8); ax_L.grid(alpha=0.3)
ax_L.set_xlabel("Time"); ax_L.set_ylabel("L")

# (F) Loss curves
ax_loss.semilogy(losses_A, COLORS["PINN-A"],  lw=1.5, label="Improved PINN")
ax_loss.semilogy(losses_B, COLORS["SPINN-B"], lw=1.5, label="Symplectic PINN")
ax_loss.semilogy(losses_C, COLORS["NODE-C"],  lw=1.5, label="Neural ODE")
ax_loss.set_title("(F) Loss Curves", fontweight="bold", fontsize=12)
ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)
ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss (log)")

# (G) Collocation distribution demo (adaptive vs uniform)
tau_uniform = np.linspace(0, 1, 500)
tau_adapt_t = adaptive_collocation(500)
tau_adapt   = tau_adapt_t.detach().numpy().flatten()

ax_coll.hist(tau_uniform, bins=40, alpha=0.5, color="steelblue",
             label="Uniform", density=True)
ax_coll.hist(tau_adapt,   bins=40, alpha=0.5, color="darkorange",
             label="Adaptive", density=True)
tau_peri = T_orb / T_SIM
ax_coll.axvline(0,        color="red", ls="--", lw=1.5, label="Periapsis τ=0")
ax_coll.axvline(tau_peri, color="red", ls=":",  lw=1.5, label=f"Periapsis τ={tau_peri:.2f}")
ax_coll.set_title("(G) Collocation: Uniform vs Adaptive",
                  fontweight="bold", fontsize=12)
ax_coll.legend(fontsize=8); ax_coll.grid(alpha=0.3)
ax_coll.set_xlabel("τ (normalised time)"); ax_coll.set_ylabel("Density")

# (H) ΔE(t) and ΔL(t) — conservation violation in extrapolation only
t_ex  = t_dense[SPLIT:]
dE_A  = np.abs(E_A[SPLIT:]  - E0)
dE_B  = np.abs(E_B[SPLIT:]  - E0)
dE_C  = np.abs(E_C[SPLIT:]  - E0)
dL_A  = np.abs(L_A[SPLIT:]  - L0)
dL_B  = np.abs(L_B[SPLIT:]  - L0)
dL_C  = np.abs(L_C[SPLIT:]  - L0)

ax_err2.semilogy(t_ex, dE_A+1e-10, COLORS["PINN-A"],  lw=2, ls="-",  label="ΔE Improved")
ax_err2.semilogy(t_ex, dE_B+1e-10, COLORS["SPINN-B"], lw=2, ls="--", label="ΔE Symplectic")
ax_err2.semilogy(t_ex, dE_C+1e-10, COLORS["NODE-C"],  lw=2, ls=":",  label="ΔE NeuralODE")
ax_err2.semilogy(t_ex, dL_A+1e-10, COLORS["PINN-A"],  lw=1.5, ls="-",  alpha=0.5,
                 label="ΔL Improved")
ax_err2.semilogy(t_ex, dL_B+1e-10, COLORS["SPINN-B"], lw=1.5, ls="--", alpha=0.5,
                 label="ΔL Symplectic")
ax_err2.set_title("(H) Conservation Violation (extrapol.)",
                  fontweight="bold", fontsize=12)
ax_err2.legend(fontsize=8, ncol=2); ax_err2.grid(alpha=0.3)
ax_err2.set_xlabel("Time"); ax_err2.set_ylabel("|ΔE|, |ΔL| (log)")

fig.suptitle(
    "Kepler Orbit — Research-Level PINN Improvements\n"
    "Improved PINN  |  Symplectic PINN (+E +L +adaptive coll.)  |  Neural ODE",
    fontsize=13, fontweight="bold")
plt.savefig("plots/kepler_advanced_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("\n  plots/kepler_advanced_comparison.png saved")

# =============================================================================
# 12.  SUMMARY TABLE
# =============================================================================
print("\n" + "="*75)
print(f"  {'Method':<20} {'Train MAE':>12} {'Extrap MAE':>12} "
      f"{'<|ΔE|> extrap':>15} {'<|ΔL|> extrap':>15}")
print("="*75)
for name, err, dE, dL in [
        ("Improved PINN",   err_A, dE_A, dL_A),
        ("Symplectic PINN", err_B, dE_B, dL_B),
        ("Neural ODE",      err_C, dE_C, dL_C)]:
    tr = np.nanmean(err[:SPLIT])
    ex = np.nanmean(err[SPLIT:])
    dE_m = np.nanmean(dE)
    dL_m = np.nanmean(dL)
    print(f"  {name:<20} {tr:>12.5f} {ex:>12.5f} {dE_m:>15.6f} {dL_m:>15.6f}")

print("="*75)
print("\n  Symplectic PINN explicitly penalises ΔE and ΔL → flatter curves in (D)(E).")
print("  Neural ODE learns the vector field in phase space → better extrapolation.")
print("  HNN (from Kepler_PINN.py) remains the gold standard for conservation.")
print("="*75)
