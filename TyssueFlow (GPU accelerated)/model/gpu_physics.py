import numpy as np
try:
    import cupy as cp
except ImportError:
    pass

def compute_mechanics_and_scan_gpu(mesh, p_config, t1_config):
    """
    Vectorized GPU Kernel for calculating Vertex Model forces and scanning for T1 transitions.
    Uses CuPy in-place scatter accumulation to bypass slow graph-traversal loops.
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
    lengths = cp.maximum(cp.sqrt(dx**2 + dy**2), 1e-6)

    # 3. T1 MASK SCANNER (Parallel threshold verification)
    e_indices = cp.arange(mesh.num_edges)
    short_mask = (lengths < t1_config.threshold) & (e_twin != -1) & (e_indices < e_twin)
    
    # 4. FACE GEOMETRIES (Vectorized shoelace formula via CuPy array scatter)
    valid_edges = cp.where(e_face != -1)[0]
    face_areas = cp.zeros(mesh.num_faces, dtype=cp.float64)
    face_perims = cp.zeros(mesh.num_faces, dtype=cp.float64)

    x1, y1 = vx[v_i[valid_edges]], vy[v_i[valid_edges]]
    x2, y2 = vx[v_next[valid_edges]], vy[v_next[valid_edges]]

    # FIXED: Corrected syntax to use CuPy's unbuffered in-place parallel accumulation methods
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
    len1 = cp.maximum(cp.sqrt(dx1**2 + dy1**2), 1e-6)

    # Combine Line Tension, Area Pressure, and Perimeter Contractility
    Fx = 0.5 * p_config.Lambda * (dx / lengths) + \
         edge_press * 0.5 * (vy[v_next] - vy[v_prev]) + \
         edge_contr * ((dx1/len1) - (dx / lengths))

    Fy = 0.5 * p_config.Lambda * (dy / lengths) + \
         edge_press * 0.5 * (vx[v_prev] - vx[v_next]) + \
         edge_contr * ((dy1/len1) - (dy / lengths))

    total_Fx = cp.zeros(mesh.num_verts, dtype=cp.float64)
    total_Fy = cp.zeros(mesh.num_verts, dtype=cp.float64)

    # Scatter force components back to vertices seamlessly across CUDA threads
    cp.add.at(total_Fx, v_i, Fx)
    cp.add.at(total_Fy, v_i, Fy)
    
    cp.add.at(total_Fx, v_next, -0.5 * p_config.Lambda * (dx / lengths))
    cp.add.at(total_Fy, v_next, -0.5 * p_config.Lambda * (dy / lengths))

    # Boundary conditions enforcement
    is_bound = cp.asarray(mesh.is_boundary_vert[:mesh.num_verts])
    total_Fx = cp.where(is_bound, 0.0, total_Fx)
    total_Fy = cp.where(is_bound, 0.0, total_Fy)

    # 7. EXPORT POINTER DATA SAFELY BACK TO CPU ARRAYS
    mesh.force_x[:mesh.num_verts] = cp.asnumpy(total_Fx)
    mesh.force_y[:mesh.num_verts] = cp.asnumpy(total_Fy)
    mesh.face_calc_area[:mesh.num_faces] = cp.asnumpy(face_areas)
    mesh.face_calc_perimeter[:mesh.num_faces] = cp.asnumpy(face_perims)

    return cp.asnumpy(cp.where(short_mask)[0])