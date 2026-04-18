"""
Test: gradient descent from random initial placement.
Does random init + gradient beat initial.plc on ibm01?
"""
import time
import numpy as np
import torch
import torch.optim as optim
import os

os.environ.setdefault("PLACE_TIME_BUDGET", "300")

from macro_place import load_benchmark

benchmark, _ = load_benchmark(
    "external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt",
    "external/MacroPlacement/Testcases/ICCAD04/ibm01/initial.plc",
    name="ibm01"
)
num_hard  = benchmark.num_hard_macros
canvas_w  = float(benchmark.canvas_width)
canvas_h  = float(benchmark.canvas_height)
grid_rows = benchmark.grid_rows
grid_cols = benchmark.grid_cols
sizes_np  = benchmark.macro_sizes.numpy().astype(np.float64)
port_pos  = benchmark.port_positions.numpy().astype(np.float64)
fixed_np  = benchmark.macro_fixed.numpy()

movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
movable_idx  = torch.where(movable_mask)[0].tolist()
mov_t        = torch.tensor(movable_idx, dtype=torch.long)

nets_np = [n.numpy() for n in benchmark.net_nodes]
hpwl_norm = max(1.0, len(nets_np) * (canvas_w + canvas_h))
n_nets  = len(nets_np)
max_nsz = max(len(n) for n in nets_np)
nnp     = np.full((n_nets, max_nsz), -1, dtype=np.int64)
for i, n in enumerate(nets_np):
    nnp[i, :len(n)] = n
nnmask_np   = (nnp >= 0)
safe_nnp_np = np.maximum(nnp, 0)
nnmask_t    = torch.tensor(nnmask_np,   dtype=torch.bool)
safe_nnp_t  = torch.tensor(safe_nnp_np, dtype=torch.long)
sizes_t     = torch.tensor(sizes_np[:num_hard], dtype=torch.double)
port_t      = torch.tensor(port_pos,            dtype=torch.double)

cell_w = canvas_w / grid_cols; cell_h = canvas_h / grid_rows
cell_cx = torch.arange(grid_cols, dtype=torch.double) * cell_w + cell_w / 2
cell_cy = torch.arange(grid_rows, dtype=torch.double) * cell_h + cell_h / 2
cell_cx_g = cell_cx.unsqueeze(0).expand(grid_rows, -1)
cell_cy_g = cell_cy.unsqueeze(1).expand(-1, grid_cols)

print(f"ibm01: {num_hard} macros, {n_nets} nets, canvas {canvas_w:.1f}x{canvas_h:.1f}")
print(f"Grid: {grid_rows}x{grid_cols}, movable: {len(movable_idx)}")

# Time one gradient step
def smooth_cost(fp, alpha=8.0):
    ap   = torch.cat([fp, port_t], dim=0)
    xs   = ap[safe_nnp_t, 0]; ys = ap[safe_nnp_t, 1]
    ni   = xs.new_full(xs.shape, -1e9)
    wl   = (torch.logsumexp(alpha * torch.where(nnmask_t, xs,  ni), 1)
          + torch.logsumexp(alpha * torch.where(nnmask_t, -xs, ni), 1)
          + torch.logsumexp(alpha * torch.where(nnmask_t, ys,  ni), 1)
          + torch.logsumexp(alpha * torch.where(nnmask_t, -ys, ni), 1)
          ).sum() / (alpha * hpwl_norm)

    cx = fp[:num_hard, 0]; cy = fp[:num_hard, 1]
    macro_w = sizes_t[:, 0]; macro_h = sizes_t[:, 1]
    sigma_x = macro_w / 2 + cell_w / 2
    sigma_y = macro_h / 2 + cell_h / 2
    dx = (cx.view(-1,1,1) - cell_cx_g.unsqueeze(0)) / sigma_x.view(-1,1,1)
    dy = (cy.view(-1,1,1) - cell_cy_g.unsqueeze(0)) / sigma_y.view(-1,1,1)
    kx = (1.0 - dx.abs()).clamp(0, 1)
    ky = (1.0 - dy.abs()).clamp(0, 1)
    kernel = kx * ky
    k_sum  = kernel.sum(dim=(1,2), keepdim=True).clamp(min=1e-9)
    density = (kernel / k_sum * (macro_w * macro_h / (cell_w * cell_h)).view(-1,1,1)).sum(0)
    dens_penalty = (density - 0.5).clamp(min=0).pow(2).mean()

    dxov = (macro_w.unsqueeze(0)+macro_w.unsqueeze(1))/2 - (cx.unsqueeze(0)-cx.unsqueeze(1)).abs()
    dyov = (macro_h.unsqueeze(0)+macro_h.unsqueeze(1))/2 - (cy.unsqueeze(0)-cy.unsqueeze(1)).abs()
    ov   = dxov.clamp(min=0) * dyov.clamp(min=0)
    ov_pen = ((ov.sum() - ov.diagonal().sum()) / 2) / (canvas_w * canvas_h)

    return wl, dens_penalty, ov_pen

# Random start
pos_rand = np.zeros((num_hard + len(port_pos), 2))
for i in movable_idx:
    w, h = sizes_np[i]
    pos_rand[i, 0] = np.random.uniform(w/2, canvas_w - w/2)
    pos_rand[i, 1] = np.random.uniform(h/2, canvas_h - h/2)
# Fixed macros
for i in range(num_hard):
    if fixed_np[i]:
        pos_rand[i] = benchmark.macro_positions.numpy()[i]
pos_rand[num_hard:] = port_pos

all_pos_t = torch.tensor(pos_rand, dtype=torch.double)
params = all_pos_t[mov_t].clone().requires_grad_(True)
optimizer = optim.Adam([params], lr=0.02)

bounds_lo_x = torch.tensor([sizes_np[i, 0]/2 for i in movable_idx], dtype=torch.double)
bounds_hi_x = torch.tensor([canvas_w - sizes_np[i, 0]/2 for i in movable_idx], dtype=torch.double)
bounds_lo_y = torch.tensor([sizes_np[i, 1]/2 for i in movable_idx], dtype=torch.double)
bounds_hi_y = torch.tensor([canvas_h - sizes_np[i, 1]/2 for i in movable_idx], dtype=torch.double)

# Time 10 steps
t0 = time.time()
for step in range(10):
    optimizer.zero_grad()
    fp = all_pos_t.clone()
    fp = fp.index_put((mov_t,), params)
    wl, dens, ov = smooth_cost(fp, alpha=2.0)  # low alpha for global opt
    cost = wl + 2.0 * dens + 500.0 * ov
    cost.backward()
    optimizer.step()
    with torch.no_grad():
        params[:, 0].clamp_(bounds_lo_x, bounds_hi_x)
        params[:, 1].clamp_(bounds_lo_y, bounds_hi_y)
step_time = (time.time() - t0) / 10
print(f"Step time: {step_time*1000:.1f}ms  |  initial cost: wl={float(wl):.4f} dens={float(dens):.4f} ov={float(ov):.6f}")

# Run for budget
budget = 180  # seconds
n_steps_expected = int(budget / step_time)
print(f"Expected steps in {budget}s: {n_steps_expected}")
alpha_schedule = [(0.2, 2.0), (0.5, 4.0), (0.8, 8.0), (1.0, 12.0)]  # (frac, alpha)

t_run = time.time()
step = 10
best_cost = float('inf'); best_params = params.detach().clone()
while time.time() - t_run < budget:
    frac = min(1.0, (time.time() - t_run) / budget)
    # Continuation: increase alpha and overlap weight over time
    alpha = 2.0 + frac * 10.0   # 2→12
    ov_w  = 50.0 + frac * 450.0  # 50→500
    dens_thresh = 0.9 - frac * 0.3  # 0.9→0.6

    optimizer.zero_grad()
    fp = all_pos_t.clone()
    fp = fp.index_put((mov_t,), params)
    ap   = torch.cat([fp, port_t], dim=0)
    xs   = ap[safe_nnp_t, 0]; ys = ap[safe_nnp_t, 1]
    ni   = xs.new_full(xs.shape, -1e9)
    wl_t = (torch.logsumexp(alpha * torch.where(nnmask_t, xs,  ni), 1)
          + torch.logsumexp(alpha * torch.where(nnmask_t, -xs, ni), 1)
          + torch.logsumexp(alpha * torch.where(nnmask_t, ys,  ni), 1)
          + torch.logsumexp(alpha * torch.where(nnmask_t, -ys, ni), 1)
          ).sum() / (alpha * hpwl_norm)
    cx = fp[:num_hard, 0]; cy = fp[:num_hard, 1]
    macro_w = sizes_t[:, 0]; macro_h = sizes_t[:, 1]
    sigma_x = macro_w / 2 + cell_w / 2; sigma_y = macro_h / 2 + cell_h / 2
    dx = (cx.view(-1,1,1) - cell_cx_g.unsqueeze(0)) / sigma_x.view(-1,1,1)
    dy = (cy.view(-1,1,1) - cell_cy_g.unsqueeze(0)) / sigma_y.view(-1,1,1)
    kernel = (1.0-dx.abs()).clamp(0,1) * (1.0-dy.abs()).clamp(0,1)
    k_sum  = kernel.sum(dim=(1,2), keepdim=True).clamp(min=1e-9)
    density = (kernel / k_sum * (macro_w * macro_h / (cell_w * cell_h)).view(-1,1,1)).sum(0)
    dens_pen = (density - dens_thresh).clamp(min=0).pow(2).mean()
    dxov = (macro_w.unsqueeze(0)+macro_w.unsqueeze(1))/2 - (cx.unsqueeze(0)-cx.unsqueeze(1)).abs()
    dyov = (macro_h.unsqueeze(0)+macro_h.unsqueeze(1))/2 - (cy.unsqueeze(0)-cy.unsqueeze(1)).abs()
    ov_t = ((dxov.clamp(0)*dyov.clamp(0)).sum() - dxov.clamp(0).diagonal()*dyov.clamp(0).diagonal()).sum() / 2
    ov_pen = ov_t / (canvas_w * canvas_h)
    cost = wl_t + 2.0 * dens_pen + ov_w * ov_pen
    cost.backward()
    optimizer.step()
    with torch.no_grad():
        params[:, 0].clamp_(bounds_lo_x, bounds_hi_x)
        params[:, 1].clamp_(bounds_lo_y, bounds_hi_y)
    c = float(wl_t.detach())
    if c < best_cost: best_cost = c; best_params = params.detach().clone()
    step += 1
    if step % 500 == 0:
        print(f"  step={step} wl={float(wl_t):.4f} dens={float(dens_pen):.4f} ov={float(ov_pen):.6f} alpha={alpha:.1f}")

print(f"\nTotal steps: {step}, best WL norm: {best_cost:.4f}")

# Now evaluate with PlacementCost
result_pos = benchmark.macro_positions.numpy().astype(np.float64).copy()
result_pos[movable_idx] = best_params.numpy()
# Quick overlap check
from macro_place._plc import PlacementCost
plc = PlacementCost("external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt")
plc.restore_placement("external/MacroPlacement/Testcases/ICCAD04/ibm01/initial.plc", ifInital=True, ifReadComment=True)
from macro_place.objective import compute_proxy_cost
r = compute_proxy_cost(torch.from_numpy(result_pos).float(), benchmark, plc)
print(f"\nGradient (random start) proxy_cost: {r['proxy_cost']:.4f}")
print(f"  wl={r['wirelength']:.3f} den={r['density']:.3f} cong={r['congestion']:.3f}")
