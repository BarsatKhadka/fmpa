import torch

from macro_place.benchmark import Benchmark


class Placer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:

        # --- Canvas ---
        canvas_w = benchmark.canvas_width          # chip width in microns
        canvas_h = benchmark.canvas_height         # chip height in microns
        canvas_area = canvas_w * canvas_h          # total chip area (μm²)

        # --- Counts ---
        num_macros = benchmark.num_macros          # total macros (hard + soft)
        num_hard   = benchmark.num_hard_macros     # hard macros: primary targets (indices 0..num_hard-1)
        num_soft   = benchmark.num_soft_macros     # soft macros: std-cell clusters (indices num_hard..end)
        num_nets   = benchmark.num_nets            # number of nets in the hypergraph

        # --- Masks ([num_macros] bool tensors) ---
        hard_mask          = benchmark.get_hard_macro_mask()              # True for hard macros
        soft_mask          = benchmark.get_soft_macro_mask()              # True for soft macros
        movable_mask       = benchmark.get_movable_mask()                 # True for non-fixed macros
        fixed_mask         = benchmark.macro_fixed                        # True for fixed macros
        movable_hard_mask  = movable_mask & hard_mask                     # movable hard macros only

        # --- Indices (integer lists / tensors) ---
        movable_hard_idx = torch.where(movable_hard_mask)[0]   # indices of movable hard macros
        fixed_idx        = torch.where(fixed_mask)[0]          # indices of fixed macros

        # --- Positions & Sizes ---
        positions = benchmark.macro_positions.clone()  # [num_macros, 2] center (x, y) in microns — THIS is what you return
        sizes     = benchmark.macro_sizes              # [num_macros, 2] (width, height) in microns

        widths  = sizes[:, 0]    # [num_macros] widths
        heights = sizes[:, 1]    # [num_macros] heights
        half_w  = widths  / 2   # half-widths  (center offset from left edge)
        half_h  = heights / 2   # half-heights (center offset from bottom edge)
        areas   = widths * heights  # [num_macros] area per macro (μm²)

        # Valid center-coordinate ranges (macro stays fully inside canvas)
        x_min = half_w   # [num_macros] minimum valid center x
        x_max = canvas_w - half_w   # [num_macros] maximum valid center x
        y_min = half_h   # [num_macros] minimum valid center y
        y_max = canvas_h - half_h   # [num_macros] maximum valid center y

        # --- Area statistics ---
        total_hard_area  = areas[:num_hard].sum().item()    # total area of all hard macros (μm²)
        utilization      = total_hard_area / canvas_area    # area utilization ratio (0–1)

        # --- Nets (hypergraph) ---
        net_nodes   = benchmark.net_nodes    # List[Tensor]: net_nodes[i] = node indices in net i
        net_weights = benchmark.net_weights  # [num_nets] float, weight per net (usually 1.0)

        # --- Grid (used by density & congestion metrics) ---
        grid_rows = benchmark.grid_rows   # number of grid rows
        grid_cols = benchmark.grid_cols   # number of grid columns
        cell_w    = canvas_w / grid_cols  # grid cell width  (μm)
        cell_h    = canvas_h / grid_rows  # grid cell height (μm)

        # --- Routing capacity ---
        hroutes = benchmark.hroutes_per_micron   # horizontal routing tracks per μm
        vroutes = benchmark.vroutes_per_micron   # vertical routing tracks per μm

        # --- I/O ports (fixed pins on chip boundary) ---
        port_positions = benchmark.port_positions   # [num_ports, 2] (x, y) in microns

        # --- Hard macro pin offsets (relative to macro center) ---
        pin_offsets = benchmark.macro_pin_offsets   # List[Tensor[num_pins, 2]], one entry per hard macro

        # --- Names (for debugging) ---
        macro_names = benchmark.macro_names   # List[str], one name per macro

        # ----------------------------------------------------------------
        # YOUR ALGORITHM GOES HERE
        # Modify `positions` for indices in movable_hard_idx.
        # Rules:
        #   - Return positions [num_macros, 2] (center coords, microns)
        #   - Hard macros must not overlap each other
        #   - All macros must stay within canvas bounds
        #   - Fixed macros must stay at their original positions
        # ----------------------------------------------------------------

        gap = 0.001
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0

        indices = sorted(movable_hard_idx.tolist(), key=lambda i: -heights[i].item())

        for idx in indices:
            w = widths[idx].item()
            h = heights[idx].item()

            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0

            if cursor_y + h > canvas_h:
                positions[idx, 0] = half_w[idx]
                positions[idx, 1] = half_h[idx]
                continue

            positions[idx, 0] = cursor_x + w / 2
            positions[idx, 1] = cursor_y + h / 2
            cursor_x += w + gap
            row_height = max(row_height, h)

        return positions
