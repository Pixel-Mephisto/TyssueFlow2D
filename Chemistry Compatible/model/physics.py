import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _numba_geometry_and_physics(
    num_verts, num_edges, num_faces,
    vert_x, vert_y, edge_srce, edge_trgt, edge_face, edge_prev, edge_next,
    face_target_area, face_calc_area, face_calc_perimeter, is_boundary_vert,
    K_A, Gamma_P, p_0, Lambda
):
    f_line_x = np.zeros(len(vert_x), dtype=np.float64)
    f_line_y = np.zeros(len(vert_x), dtype=np.float64)
    f_area_x = np.zeros(len(vert_x), dtype=np.float64)
    f_area_y = np.zeros(len(vert_x), dtype=np.float64)
    f_perim_x = np.zeros(len(vert_x), dtype=np.float64)
    f_perim_y = np.zeros(len(vert_x), dtype=np.float64)

    for f in prange(num_faces):
        start_edge = -1
        for e in range(num_edges):
            if edge_face[e] == f:
                start_edge = e
                break
        if start_edge != -1:
            curr_edge = start_edge
            area_sum = 0.0
            perimeter_sum = 0.0
            while True:
                v1 = edge_srce[curr_edge]
                v2 = edge_trgt[curr_edge]
                x1, y1 = vert_x[v1], vert_y[v1]
                x2, y2 = vert_x[v2], vert_y[v2]
                area_sum += (x1 * y2 - x2 * y1)
                perimeter_sum += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                curr_edge = edge_next[curr_edge]
                if curr_edge == start_edge: break
            face_calc_area[f] = 0.5 * np.abs(area_sum)
            face_calc_perimeter[f] = perimeter_sum

    for e in prange(num_edges):
        v_i = edge_srce[e]
        v_next = edge_trgt[e]
        f = edge_face[e]
        
        dx = vert_x[v_next] - vert_x[v_i]
        dy = vert_y[v_next] - vert_y[v_i]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 1e-6:
            fx = 0.5 * Lambda * (dx / length)
            fy = 0.5 * Lambda * (dy / length)
            f_line_x[v_i] += fx; f_line_y[v_i] += fy
            f_line_x[v_next] -= fx; f_line_y[v_next] -= fy

        if f != -1:
            eprev = edge_prev[e]
            v_prev = edge_srce[eprev]
            A_c = face_calc_area[f]
            P_c = face_calc_perimeter[f]
            A_0 = face_target_area[f]
            P_0 = p_0 * np.sqrt(A_0)
            
            pressure = -K_A * (A_c - A_0)
            contractility = -Gamma_P * (P_c - P_0)
            
            dx1, dy1 = vert_x[v_i] - vert_x[v_prev], vert_y[v_i] - vert_y[v_prev]
            dx2, dy2 = vert_x[v_next] - vert_x[v_i], vert_y[v_next] - vert_y[v_i]
            len1, len2 = np.sqrt(dx1*dx1 + dy1*dy1), np.sqrt(dx2*dx2 + dy2*dy2)
            
            if len1 > 1e-6 and len2 > 1e-6:
                f_perim_x[v_i] += contractility * ((dx1 / len1) - (dx2 / len2))
                f_perim_y[v_i] += contractility * ((dy1 / len1) - (dy2 / len2))
            f_area_x[v_i] += pressure * (0.5 * (vert_y[v_next] - vert_y[v_prev]))
            f_area_y[v_i] += pressure * (0.5 * (vert_x[v_prev] - vert_x[v_next]))

    total_fx = np.zeros(len(vert_x), dtype=np.float64)
    total_fy = np.zeros(len(vert_x), dtype=np.float64)
    for v in prange(num_verts):
        if not is_boundary_vert[v]:
            total_fx[v] = f_line_x[v] + f_area_x[v] + f_perim_x[v]
            total_fy[v] = f_line_y[v] + f_area_y[v] + f_perim_y[v]

    return total_fx, total_fy

@njit
def _compute_system_energy(num_edges, num_faces, edge_srce, edge_trgt, vert_x, vert_y, face_calc_area, face_calc_perimeter, face_target_area, K_A, Gamma_P, p_0, Lambda):
    """Calculates Hamiltonian state directly from arrays for high-speed tracking."""
    E_total = 0.0
    for e in range(num_edges):
        v_i, v_next = edge_srce[e], edge_trgt[e]
        dx, dy = vert_x[v_next] - vert_x[v_i], vert_y[v_next] - vert_y[v_i]
        E_total += 0.5 * Lambda * np.sqrt(dx*dx + dy*dy)
        
    for f in range(num_faces):
        A_c = face_calc_area[f]
        P_c = face_calc_perimeter[f]
        A_0 = face_target_area[f]
        E_total += 0.5 * K_A * (A_c - A_0)**2
        E_total += 0.5 * Gamma_P * (P_c - p_0 * np.sqrt(A_0))**2
        
    return E_total

@njit
def resolve_t1_fast_numba(num_edges, vert_x, vert_y, edge_srce, edge_trgt, edge_twin, edge_next, edge_prev, edge_face, threshold, rest_length):
    swaps = 0
    for e in range(num_edges):
        twin = edge_twin[e]
        
        # 1. Protection: Skip boundary edges and prevent double-swapping the same pair
        if twin == -1: continue
        if e > twin: continue 

        v1, v2 = edge_srce[e], edge_trgt[e]
        dx, dy = vert_x[v2] - vert_x[v1], vert_y[v2] - vert_y[v1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < threshold:
            # --- EXTRACT SURROUNDING TOPOLOGY ---
            he1_prev, he1_next = edge_prev[e], edge_next[e]
            he2_prev, he2_next = edge_prev[twin], edge_next[twin]
            
            he1_prev_twin = edge_twin[he1_prev]
            he1_next_twin = edge_twin[he1_next]
            he2_prev_twin = edge_twin[he2_prev]
            he2_next_twin = edge_twin[he2_next]
            
            # Abort if the swap borders the void (prevents mesh destruction)
            if -1 in [he1_prev_twin, he1_next_twin, he2_prev_twin, he2_next_twin]: 
                continue
                
            f3 = edge_face[he1_prev_twin]
            f4 = edge_face[he1_next_twin]
            
            # --- 1. GEOMETRIC FLIP ---
            cx, cy = (vert_x[v1] + vert_x[v2]) / 2.0, (vert_y[v1] + vert_y[v2]) / 2.0
            nx, ny = -dy / length, dx / length
            
            vert_x[v1], vert_y[v1] = cx + nx * rest_length / 2.0, cy + ny * rest_length / 2.0
            vert_x[v2], vert_y[v2] = cx - nx * rest_length / 2.0, cy - ny * rest_length / 2.0
            
            # --- 2. EXACT TOPOLOGICAL REWIRING ---
            
            # Update vertex ownership for the shifted 'next' edges
            edge_srce[he1_next] = v1; edge_trgt[he1_next_twin] = v1
            edge_srce[he2_next] = v2; edge_trgt[he2_next_twin] = v2
            
            # Close the original shrinking faces
            edge_next[he1_prev] = he1_next; edge_prev[he1_next] = he1_prev
            edge_next[he2_prev] = he2_next; edge_prev[he2_next] = he2_prev
            
            # Splice Half-Edge 1 into the Top Face (f4)
            edge_face[e] = f4
            edge_next[he1_next_twin] = e; edge_prev[e] = he1_next_twin
            edge_next[e] = he2_prev_twin; edge_prev[he2_prev_twin] = e
            
            # Splice Half-Edge 2 (twin) into the Bottom Face (f3)
            edge_face[twin] = f3
            edge_next[he2_next_twin] = twin; edge_prev[twin] = he2_next_twin
            edge_next[twin] = he1_prev_twin; edge_prev[he1_prev_twin] = twin
            
            swaps += 1
            
    return swaps

def run_physics_step(mesh, p_config):
    """Executes geometry updates, applies forces, and returns scalar energy."""
    fx, fy = _numba_geometry_and_physics(
        mesh.num_verts, mesh.num_edges, mesh.num_faces,
        mesh.vert_x, mesh.vert_y, mesh.edge_srce, mesh.edge_trgt, mesh.edge_face, mesh.edge_prev, mesh.edge_next,
        mesh.face_target_area, mesh.face_calc_area, mesh.face_calc_perimeter, mesh.is_boundary_vert,
        p_config.K_A, p_config.Gamma_P, p_config.p_0, p_config.Lambda
    )
    mesh.force_x[:mesh.num_verts] = fx[:mesh.num_verts]
    mesh.force_y[:mesh.num_verts] = fy[:mesh.num_verts]
    
    energy = _compute_system_energy(
        mesh.num_edges, mesh.num_faces, mesh.edge_srce, mesh.edge_trgt,
        mesh.vert_x, mesh.vert_y, mesh.face_calc_area, mesh.face_calc_perimeter, mesh.face_target_area,
        p_config.K_A, p_config.Gamma_P, p_config.p_0, p_config.Lambda
    )
    return energy