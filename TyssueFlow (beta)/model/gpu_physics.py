import numpy as np
try:
    import cupy as cp
except ImportError:
    pass

def compute_mechanics_and_scan_gpu(mesh, p_config, t1_config):
    """
    Vectorized GPU Kernel for calculating Vertex Model forces and scanning for T1 transitions.
    Uses CuPy in-place scatter accumulation to bypass slow graph-traversal loops.
    Includes Minimum Image Convention wrapping for Periodic Boundary Conditions.
    Protects frozen periodic boundary edges from destructive T1 topology swaps.
    """
    # 1. PUSH UNWRAPPED FLAT DATA TO GPU VRAM
    vx = cp.asarray(mesh.vert_x[:mesh.num_verts])
    vy = cp.asarray(mesh.vert_y[:mesh.num_verts])
    v_i = cp.asarray(mesh.edge_srce[:mesh.num_edges])
    v_next = cp.asarray(mesh.edge_trgt[:mesh.num_edges])
    e_face = cp.asarray(mesh.edge_face[:mesh.num_edges])
    e_prev = cp.asarray(mesh.edge_prev[:mesh.num_edges])
    e_twin = cp.asarray(mesh.edge_twin[:mesh.num_edges])

    # 2. COMPUTE ALL EDGE VECTOR GEOMETRIES IN PARALLEL
    dx = vx[v_next] - vx[v_i]
    dy = vy[v_next] - vy[v_i]

    # Apply Minimum Image Convention for Periodic Boundary Conditions
    if getattr(mesh, 'use_pbc', False):
        Lx = mesh.exact_Lx
        Ly = mesh.exact_Ly
        dx = dx - Lx * cp.round(dx / Lx)
        dy = dy - Ly * cp.round(dy / Ly)

    lengths = cp.maximum(cp.sqrt(dx**2 + dy**2), 1e-6)

    # 3. T1 MASK SCANNER (Parallel threshold verification & boundary shielding)
    e_indices = cp.arange(mesh.num_edges)
    short_mask = (lengths < t1_config.threshold) & (e_twin != -1) & (e_indices < e_twin)
    
    if getattr(mesh, 'use_pbc', False):
        # Identify edges wrapping across the periodic macro boundaries
        # Look up the true source vertex ID of the twin edge securely
        v_twin_srce = cp.where(e_twin == -1, v_i, cp.asarray(mesh.edge_srce)[e_twin])
        
        # Calculate raw unwrapped coordinate distances across VRAM tensors
        raw_dx = cp.abs(vx[v_i] - vx[v_twin_srce])
        raw_dy = cp.abs(vy[v_i] - vy[v_twin_srce])
        
        # Flag macro-wrapping boundary edges
        is_pbc_boundary_edge = (raw_dx > 0.5 * mesh.exact_Lx) | (raw_dy > 0.5 * mesh.exact_Ly)
        
        # Stripping out boundary edges from triggering topology swaps
        short_mask = short_mask & (~is_pbc_boundary_edge)

    # 4. FACE GEOMETRIES (Vectorized shoelace formula via CuPy array scatter)
    valid_edges = cp.where(e_face != -1)[0]
    face_areas = cp.zeros(mesh.num_faces, dtype=cp.float64)
    face_perims = cp.zeros(mesh.num_faces, dtype=cp.float64)

    # In PBC mode, we use relative position tracking to ensure shoelace area calculations
    # do not explode when crossing boundaries
    if getattr(mesh, 'use_pbc', False):
        x1 = vx[v_i[valid_edges]]
        y1 = vy[v_i[valid_edges]]
        x2 = x1 + dx[valid_edges]
        y2 = y1 + dy[valid_edges]
    else:
        x1, y1 = vx[v_i[valid_edges]], vy[v_i[valid_edges]]
        x2, y2 = vx[v_next[valid_edges]], vy[v_next[valid_edges]]

    # CuPy's unbuffered in-place parallel accumulation methods
    cp.add.at(face_areas, e_face[valid_edges], x1*y2 - x2*y1)
    cp.add.at(face_perims, e_face[valid_edges], lengths[valid_edges])
    face_areas = 0.5 * cp.abs(face_areas)

    # 5. PRESSURE & CONTRACTILITY CALCULATION
    A_0 = cp.asarray(mesh.face_target_area[:mesh.num_faces])
    P_0 = p_config.p_0 * cp.sqrt(A_0)

    pressure = -p_config.K_A * (face_areas - A_0)
    contractility = -p_config.Gamma_P * (face_perims - P_0)

    edge_press = cp.where(e_face == -1, 0.0, pressure[e_face])
    edge_contr = cp.where(e_face == -1, 0.0, contractility[e_face])

    # 6. VERTEX FORCES CALCULATION
    v_prev = v_i[e_prev]
    dx1 = vx[v_i] - vx[v_prev]
    dy1 = vy[v_i] - vy[v_prev]

    # Apply Minimum Image Convention for the previous adjacent edge tracking
    if getattr(mesh, 'use_pbc', False):
        Lx = mesh.exact_Lx
        Ly = mesh.exact_Ly
        dx1 = dx1 - Lx * cp.round(dx1 / Lx)
        dy1 = dy1 - Ly * cp.round(dy1 / Ly)

    len1 = cp.maximum(cp.sqrt(dx1**2 + dy1**2), 1e-6)

    # Combine Line Tension, Area Pressure, and Perimeter Contractility
    # For pressure forces, use the wrapped delta values relative to current position
    vy_next_rel = vy[v_i] + dy
    vy_prev_rel = vy[v_i] - dy1
    vx_next_rel = vx[v_i] + dx
    vx_prev_rel = vx[v_i] - dx1

    Fx = 0.5 * p_config.Lambda * (dx / lengths) + \
         edge_press * 0.5 * (vy_next_rel - vy_prev_rel) + \
         edge_contr * ((dx1/len1) - (dx / lengths))

    Fy = 0.5 * p_config.Lambda * (dy / lengths) + \
         edge_press * 0.5 * (vx_prev_rel - vx_next_rel) + \
         edge_contr * ((dy1/len1) - (dy / lengths))

    total_Fx = cp.zeros(mesh.num_verts, dtype=cp.float64)
    total_Fy = cp.zeros(mesh.num_verts, dtype=cp.float64)

    # Scatter force components back to vertices seamlessly across CUDA threads
    cp.add.at(total_Fx, v_i, Fx)
    cp.add.at(total_Fy, v_i, Fy)
    
    cp.add.at(total_Fx, v_next, -0.5 * p_config.Lambda * (dx / lengths))
    cp.add.at(total_Fy, v_next, -0.5 * p_config.Lambda * (dy / lengths))

    # Boundary conditions enforcement (perimeter cells remain mechanically frozen)
    is_bound = cp.asarray(mesh.is_boundary_vert[:mesh.num_verts])
    total_Fx = cp.where(is_bound, 0.0, total_Fx)
    total_Fy = cp.where(is_bound, 0.0, total_Fy)

    # 7. EXPORT POINTER DATA SAFELY BACK TO CPU ARRAYS
    mesh.force_x[:mesh.num_verts] = cp.asnumpy(total_Fx)
    mesh.force_y[:mesh.num_verts] = cp.asnumpy(total_Fy)
    mesh.face_calc_area[:mesh.num_faces] = cp.asnumpy(face_areas)
    mesh.face_calc_perimeter[:mesh.num_faces] = cp.asnumpy(face_perims)

    return cp.asnumpy(cp.where(short_mask)[0])