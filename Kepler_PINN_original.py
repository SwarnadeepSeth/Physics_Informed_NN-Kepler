"""
Kepler Orbit — Physics-Informed Neural Network (PINN)
======================================================
Governing equations (dimensionless units, GM=1):
    ẍ = -x / r³
    ÿ = -y / r³
    r = sqrt(x² + y²)

The PINN is trained on sparse position observations from the first part of the
orbit and must reconstruct the complete elliptical trajectory using:
  1. Data loss  — MSE between predicted and observed (x, y) positions
  2. Physics loss — mean-squared ODE residuals from Newton's gravity law

Compare against:
  - Numerical reference (scipy.integrate.odeint, Runge-Kutta)
  - Standard NN (data-only, same architecture)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image

os.makedirs("plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ORBITAL MECHANICS SETUP
# ─────────────────────────────────────────────────────────────────────────────
GM          = 1.0    # gravitational parameter (normalised)
ECCENTRICITY = 0.5   # orbital eccentricity  (0=circle, <1=ellipse)
# Semi-major axis a=1 → periapsis at x = 1-e, apoapsis at x = -(1+e)
a           = 1.0

# Initial conditions at periapsis (closest approach)
x0   = a * (1 - ECCENTRICITY)            # periapsis distance
vx0  = 0.0                                # no radial velocity at periapsis
vy0  = np.sqrt(GM * (1 + ECCENTRICITY) / (a * (1 - ECCENTRICITY)))  # vis-viva

# Orbital period via Kepler's third law: T = 2π √(a³/GM)
T_orbit = 2 * np.pi * np.sqrt(a**3 / GM)

# We simulate 1.5 orbital periods
T_SIM    = 1.5 * T_orbit
N_DENSE  = 1000      # resolution of the dense reference grid
END_TRAIN_FRAC = 0.4  # fraction of orbit used as training data

print(f"Orbital eccentricity : e = {ECCENTRICITY}")
print(f"Orbital period       : T = {T_orbit:.4f} (normalised units)")
print(f"Simulation time      : {T_SIM:.4f}")
print(f"Initial conditions   : x₀={x0:.3f}, y₀=0, vx₀={vx0:.3f}, vy₀={vy0:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. NUMERICAL SOLUTION (reference)
# ─────────────────────────────────────────────────────────────────────────────
def kepler_ode(state, t):
    """
    Two-body gravitational ODE:
        state = [x, y, vx, vy]
        ẍ = -GM·x/r³,  ÿ = -GM·y/r³
    """
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    return [vx, vy, ax, ay]


t_dense = np.linspace(0, T_SIM, N_DENSE)
state0  = [x0, 0.0, vx0, vy0]
sol     = odeint(kepler_ode, state0, t_dense, rtol=1e-10, atol=1e-12)

x_exact  = sol[:, 0]
y_exact  = sol[:, 1]
vx_exact = sol[:, 2]
vy_exact = sol[:, 3]
r_exact  = np.sqrt(x_exact**2 + y_exact**2)

# Conserved quantities: energy E and angular momentum L
E_exact = 0.5 * (vx_exact**2 + vy_exact**2) - GM / r_exact
L_exact = x_exact * vy_exact - y_exact * vx_exact

print(f"\nConservation check (should be constant):")
print(f"  Energy    range : [{E_exact.min():.6f}, {E_exact.max():.6f}]")
print(f"  Ang. mom. range : [{L_exact.min():.6f}, {L_exact.max():.6f}]")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPARE TENSORS
# ─────────────────────────────────────────────────────────────────────────────
t_tensor = torch.tensor(t_dense, dtype=torch.float32).view(-1, 1)
x_tensor = torch.tensor(x_exact, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y_exact, dtype=torch.float32).view(-1, 1)
xy_full  = torch.cat([x_tensor, y_tensor], dim=1)  # shape (N, 2)

# Sparse training data: sample from first END_TRAIN_FRAC of the orbit
train_mask  = t_dense <= END_TRAIN_FRAC * T_SIM
train_idx   = np.where(train_mask)[0][::15]          # every 15th point
t_data      = t_tensor[train_idx]
xy_data     = xy_full[train_idx]

print(f"\nTraining points : {len(train_idx)}  (from {len(t_dense)} total)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class FCN(nn.Module):
    """Fully-connected network: input=1 (time), output=2 (x, y)."""

    def __init__(self, n_hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def save_gif_PIL(outfile, files, fps=10, loop=0):
    imgs = [Image.open(f) for f in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:],
                 save_all=True, duration=int(1000 / fps), loop=loop)


def plot_orbit_snapshot(t_np, xy_exact, xy_pred, t_data_np, xy_data_np,
                        epoch, label, color, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- trajectory in (x,y) space ---
    ax = axes[0]
    ax.plot(xy_exact[:, 0], xy_exact[:, 1], color='#2ecc71', lw=2,
            label='Exact orbit', zorder=3)
    ax.plot(xy_pred[:, 0],  xy_pred[:, 1],  color=color, lw=2.5,
            ls='--', alpha=0.85, label=f'{label} prediction', zorder=4)
    ax.scatter(xy_data_np[:, 0], xy_data_np[:, 1], s=60, color='#f39c12',
               zorder=5, label='Training data', edgecolors='k', lw=0.5)
    ax.scatter([0], [0], s=200, color='gold', marker='*', zorder=6,
               label='Central body', edgecolors='orange', lw=1)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')
    ax.set_title(f'Orbital Trajectory  —  step {epoch:,}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)

    # --- x(t) and y(t) over time ---
    ax2 = axes[1]
    ax2.plot(t_np, xy_exact[:, 0], color='#2ecc71', lw=1.8, alpha=0.8, label='x exact')
    ax2.plot(t_np, xy_exact[:, 1], color='#27ae60', lw=1.8, alpha=0.8, ls='--', label='y exact')
    ax2.plot(t_np, xy_pred[:, 0],  color=color,     lw=2.5, alpha=0.85, label=f'x {label}')
    ax2.plot(t_np, xy_pred[:, 1],  color=color,     lw=2.5, alpha=0.85, ls='--', label=f'y {label}')
    ax2.axvspan(0, END_TRAIN_FRAC * T_SIM, alpha=0.08, color='orange', label='Training region')
    ax2.set_xlabel('Time [T]', fontsize=12)
    ax2.set_ylabel('Position', fontsize=12)
    ax2.set_title('Position vs Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=90, facecolor='white')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. STANDARD NN (data-only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("Training Standard Neural Network (data only) …")
print("─"*60)

torch.manual_seed(42)
nn_model  = FCN(n_hidden=64, n_layers=4)
nn_optim  = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
NN_EPOCHS = 5000
nn_losses = []

files_nn = []
for i in range(NN_EPOCHS):
    nn_optim.zero_grad()
    pred   = nn_model(t_data)
    loss   = torch.mean((pred - xy_data) ** 2)
    loss.backward()
    nn_optim.step()
    nn_losses.append(loss.item())

    if (i + 1) % 250 == 0:
        with torch.no_grad():
            xy_pred = nn_model(t_tensor).numpy()
        fn = f"plots/nn_{i+1:07d}.png"
        plot_orbit_snapshot(t_dense, xy_full.numpy(), xy_pred,
                            t_data.numpy(), xy_data.numpy(),
                            i + 1, 'NN', '#3498db', fn)
        files_nn.append(fn)
        if (i + 1) % 2500 == 0:
            print(f"  epoch {i+1:5d}  loss={loss.item():.3e}")

save_gif_PIL("nn_kepler.gif", files_nn, fps=8)
print("✅  nn_kepler.gif saved")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PINN (data + physics loss)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("Training PINN (data + Kepler ODE residual) …")
print("─"*60)

N_COLLOC     = 50
LAMBDA_PHYS  = 1e-3
PINN_EPOCHS  = 30000
PINN_LR      = 5e-4

t_physics = torch.linspace(0, T_SIM, N_COLLOC).view(-1, 1).requires_grad_(True)

torch.manual_seed(42)
pinn_model  = FCN(n_hidden=64, n_layers=4)
pinn_optim  = torch.optim.Adam(pinn_model.parameters(), lr=PINN_LR)

pinn_losses      = []
pinn_data_losses = []
pinn_phys_losses = []

files_pinn = []
for i in range(PINN_EPOCHS):
    pinn_optim.zero_grad()

    # ── Data loss ─────────────────────────────────────────────────────────────
    pred_data = pinn_model(t_data)
    loss_data = torch.mean((pred_data - xy_data) ** 2)

    # ── Physics loss ──────────────────────────────────────────────────────────
    pred_phys = pinn_model(t_physics)    # shape (M, 2)
    xp = pred_phys[:, 0:1]              # predicted x(t)
    yp = pred_phys[:, 1:2]              # predicted y(t)

    # First derivatives via autograd
    vx_p = torch.autograd.grad(xp, t_physics, torch.ones_like(xp), create_graph=True)[0]
    vy_p = torch.autograd.grad(yp, t_physics, torch.ones_like(yp), create_graph=True)[0]

    # Second derivatives (accelerations)
    ax_p = torch.autograd.grad(vx_p, t_physics, torch.ones_like(vx_p), create_graph=True)[0]
    ay_p = torch.autograd.grad(vy_p, t_physics, torch.ones_like(vy_p), create_graph=True)[0]

    # Gravitational acceleration from position
    r_p   = torch.sqrt(xp**2 + yp**2 + 1e-8)   # small eps for stability
    ax_grav = -GM * xp / r_p**3
    ay_grav = -GM * yp / r_p**3

    # ODE residuals: ẍ + GM·x/r³ = 0
    res_x = ax_p - ax_grav
    res_y = ay_p - ay_grav
    loss_physics = torch.mean(res_x**2 + res_y**2)

    # ── Combined loss ─────────────────────────────────────────────────────────
    loss = loss_data + LAMBDA_PHYS * loss_physics
    loss.backward()
    pinn_optim.step()

    pinn_losses.append(loss.item())
    pinn_data_losses.append(loss_data.item())
    pinn_phys_losses.append(loss_physics.item())

    if (i + 1) % 1000 == 0:
        with torch.no_grad():
            xy_pred = pinn_model(t_tensor).numpy()
        fn = f"plots/pinn_{i+1:07d}.png"
        plot_orbit_snapshot(t_dense, xy_full.numpy(), xy_pred,
                            t_data.numpy(), xy_data.numpy(),
                            i + 1, 'PINN', '#e74c3c', fn)
        files_pinn.append(fn)
        if (i + 1) % 5000 == 0:
            print(f"  epoch {i+1:6d}  total={loss.item():.3e}  data={loss_data.item():.3e}  "
                  f"phys={loss_physics.item():.3e}")

save_gif_PIL("pinn_kepler.gif", files_pinn, fps=8)
print("✅  pinn_kepler.gif saved")


# ─────────────────────────────────────────────────────────────────────────────
# 8. FINAL COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────
with torch.no_grad():
    xy_nn_full   = nn_model(t_tensor).numpy()
    xy_pinn_full = pinn_model(t_tensor).numpy()

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# ─── A: orbital trajectories ──────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(x_exact, y_exact,              color='#2ecc71', lw=2.5, label='Exact')
ax.plot(xy_nn_full[:, 0],   xy_nn_full[:, 1],   color='#3498db', lw=2.5, ls='-.', label='Standard NN')
ax.plot(xy_pinn_full[:, 0], xy_pinn_full[:, 1], color='#e74c3c', lw=2.5, ls='--', label='PINN')
ax.scatter([0], [0], s=300, color='gold', marker='*', zorder=6, edgecolors='orange', lw=1.5)
ax.scatter(xy_data.numpy()[:, 0], xy_data.numpy()[:, 1], s=60, color='#f39c12',
           zorder=5, edgecolors='k', lw=0.5, label='Training data')
ax.set_aspect('equal')
ax.set_title('(A) Orbital Trajectories', fontweight='bold', fontsize=12)
ax.set_xlabel('x', fontsize=11); ax.set_ylabel('y', fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)

# ─── B: x(t) comparison ───────────────────────────────────────────────────────
ax2 = axes[0, 1]
ax2.plot(t_dense, x_exact,             color='#2ecc71', lw=2.5, label='Exact')
ax2.plot(t_dense, xy_nn_full[:, 0],   color='#3498db', lw=2.5, ls='-.', label='NN')
ax2.plot(t_dense, xy_pinn_full[:, 0], color='#e74c3c', lw=2.5, ls='--', label='PINN')
ax2.axvspan(0, END_TRAIN_FRAC * T_SIM, alpha=0.07, color='orange', label='Training region')
ax2.set_title('(B) x-coordinate over Time', fontweight='bold', fontsize=12)
ax2.set_xlabel('Time', fontsize=11); ax2.set_ylabel('x', fontsize=11)
ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

# ─── C: position error ────────────────────────────────────────────────────────
ax3 = axes[1, 0]
err_nn   = np.sqrt((xy_nn_full[:, 0]   - x_exact)**2 + (xy_nn_full[:, 1]   - y_exact)**2)
err_pinn = np.sqrt((xy_pinn_full[:, 0] - x_exact)**2 + (xy_pinn_full[:, 1] - y_exact)**2)
ax3.semilogy(t_dense, err_nn   + 1e-10, color='#3498db', lw=2, label='NN error')
ax3.semilogy(t_dense, err_pinn + 1e-10, color='#e74c3c', lw=2, label='PINN error')
ax3.axvline(x=END_TRAIN_FRAC * T_SIM, color='gray', ls=':', lw=1.5, label='End of training data')
ax3.set_title('(C) Position Error |Δr|', fontweight='bold', fontsize=12)
ax3.set_xlabel('Time', fontsize=11); ax3.set_ylabel('|Δr| (log scale)', fontsize=11)
ax3.legend(fontsize=10); ax3.grid(alpha=0.3)

# ─── D: loss curves ───────────────────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.semilogy(nn_losses,        color='#3498db', lw=1.5, label='NN data loss')
ax4.semilogy(pinn_data_losses, color='#e74c3c', lw=1.5, label='PINN data loss')
ax4.semilogy([LAMBDA_PHYS * v for v in pinn_phys_losses],
             color='#9b59b6', lw=1.5, label='λ × PINN physics loss')
ax4.set_title('(D) Loss Curves', fontweight='bold', fontsize=12)
ax4.set_xlabel('Epoch', fontsize=11); ax4.set_ylabel('Loss (log)', fontsize=11)
ax4.legend(fontsize=10); ax4.grid(alpha=0.3)

plt.suptitle(r'Kepler Orbit PINN  —  $e = 0.5$ ellipse, $GM = 1$',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/kepler_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅  plots/kepler_comparison.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
split   = int(END_TRAIN_FRAC * N_DENSE)
print("\n── Error Statistics ─────────────────────────────────────────────────")
print(f"{'Method':<14}  {'Train MAE':>12}  {'Extrapol. MAE':>14}")
print(f"{'Standard NN':<14}  {err_nn[:split].mean():>12.5f}  {err_nn[split:].mean():>14.5f}")
print(f"{'PINN':<14}  {err_pinn[:split].mean():>12.5f}  {err_pinn[split:].mean():>14.5f}")
print(f"\nPINN improvement (extrapolation): {err_nn[split:].mean()/max(err_pinn[split:].mean(),1e-9):.1f}×")
