import numpy as np
import pandas as pd
from typing import Dict
from numba import njit
from mesh_engine import TissueMesh

# Removed cache=True to prevent compiler state lockups across separate simulation instances
@njit
def _numba_physics(
    edge_idxs, edge_srce, edge_trgt, edge_face, edge_prev,
    vert_x, vert_y,
    face_idxs, face_calc_area, face_calc_perim, face_target_area,
    K_A, Gamma_P, p_0, Lambda
):
    """Pre-compiled C-level loop bypassing all Pandas overhead."""
    max_vid = len(vert_x)
    f_line_x = np.zeros(max_vid, dtype=np.float64)
    f_line_y = np.zeros(max_vid, dtype=np.float64)
    f_area_x = np.zeros(max_vid, dtype=np.float64)
    f_area_y = np.zeros(max_vid, dtype=np.float64)
    f_perim_x = np.zeros(max_vid, dtype=np.float64)
    f_perim_y = np.zeros(max_vid, dtype=np.float64)

    E_line = 0.0
    E_area = 0.0
    E_perim = 0.0

    # 1. Edge-Level Mechanics
    for i in range(len(edge_idxs)):
        e_curr = edge_idxs[i]
        v_i = edge_srce[e_curr]
        v_next = edge_trgt[e_curr]
        
        # --- Line Tension ---
        dx = vert_x[v_next] - vert_x[v_i]
        dy = vert_y[v_next] - vert_y[v_i]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 1e-6:
            E_line += 0.5 * Lambda * length
            fx = 0.5 * Lambda * (dx / length)
            fy = 0.5 * Lambda * (dy / length)
            f_line_x[v_i] += fx
            f_line_y[v_i] += fy
            f_line_x[v_next] -= fx
            f_line_y[v_next] -= fy

        # --- Face Contractility & Pressure ---
        f = edge_face[e_curr]
        if f != -1: 
            e_prev = edge_prev[e_curr]
            v_prev = edge_srce[e_prev]

            A_c = face_calc_area[f]
            P_c = face_calc_perim[f]
            A_0 = face_target_area[f]
            P_0 = p_0 * np.sqrt(A_0)

            pressure = -K_A * (A_c - A_0)
            contractility = -Gamma_P * (P_c - P_0)

            xi, yi = vert_x[v_i], vert_y[v_i]
            x_next, y_next = vert_x[v_next], vert_y[v_next]
            x_prev, y_prev = vert_x[v_prev], vert_y[v_prev]

            dx1, dy1 = xi - x_prev, yi - y_prev
            dx2, dy2 = x_next - xi, y_next - yi
            len1 = np.sqrt(dx1*dx1 + dy1*dy1)
            len2 = np.sqrt(dx2*dx2 + dy2*dy2)

            if len1 > 1e-6 and len2 > 1e-6:
                f_perim_x[v_i] += contractility * ((dx1 / len1) - (dx2 / len2))
                f_perim_y[v_i] += contractility * ((dy1 / len1) - (dy2 / len2))

            f_area_x[v_i] += pressure * (0.5 * (y_next - y_prev))
            f_area_y[v_i] += pressure * (0.5 * (x_prev - x_next))

    # 2. Face-Level Energy Accumulation
    for i in range(len(face_idxs)):
        f = face_idxs[i]
        if f != -1:
            A_c = face_calc_area[f]
            P_c = face_calc_perim[f]
            A_0 = face_target_area[f]
            E_area += 0.5 * K_A * (A_c - A_0)**2
            E_perim += 0.5 * Gamma_P * (P_c - p_0 * np.sqrt(A_0))**2

    return E_line, E_area, E_perim, f_line_x, f_line_y, f_area_x, f_area_y, f_perim_x, f_perim_y


def compute_system_energy(mesh: TissueMesh, 
                          K_A: float = 1.0, 
                          Gamma_P: float = 0.1, 
                          p_0: float = 3.81,
                          Lambda: float = 1.0) -> Dict[str, float]:
    mesh.compute_geometry()
    
    max_vid = int(mesh.vert_df.index.max() + 1)
    max_eid = int(mesh.edge_df.index.max() + 1)
    max_fid = int(mesh.face_df.index.max() + 1)

    vert_x = np.zeros(max_vid, dtype=np.float64)
    vert_y = np.zeros(max_vid, dtype=np.float64)
    vert_x[mesh.vert_df.index] = mesh.vert_df['x'].values
    vert_y[mesh.vert_df.index] = mesh.vert_df['y'].values

    # Strict typecasting guarantees no cache/memory signature mismatches
    edge_idxs = mesh.edge_df.index.values.astype(np.int64)
    edge_srce = np.zeros(max_eid, dtype=np.int64)
    edge_trgt = np.zeros(max_eid, dtype=np.int64)
    edge_face = np.zeros(max_eid, dtype=np.int64)
    edge_prev = np.zeros(max_eid, dtype=np.int64)
    
    edge_srce[edge_idxs] = mesh.edge_df['srce'].values.astype(np.int64)
    edge_trgt[edge_idxs] = mesh.edge_df['trgt'].values.astype(np.int64)
    edge_face[edge_idxs] = mesh.edge_df['face'].values.astype(np.int64)
    edge_prev[edge_idxs] = mesh.edge_df['prev'].values.astype(np.int64)

    face_idxs = mesh.face_df.index.values.astype(np.int64)
    face_calc_area = np.zeros(max_fid, dtype=np.float64)
    face_calc_perim = np.zeros(max_fid, dtype=np.float64)
    face_target_area = np.zeros(max_fid, dtype=np.float64)
    
    face_calc_area[face_idxs] = mesh.face_df['calc_area'].values.astype(np.float64)
    face_calc_perim[face_idxs] = mesh.face_df['calc_perimeter'].values.astype(np.float64)
    face_target_area[face_idxs] = mesh.face_df['target_area'].values.astype(np.float64)

    res = _numba_physics(
        edge_idxs, edge_srce, edge_trgt, edge_face, edge_prev,
        vert_x, vert_y,
        face_idxs, face_calc_area, face_calc_perim, face_target_area,
        K_A, Gamma_P, p_0, Lambda
    )
    E_line, E_area, E_perim, flx, fly, fax, fay, fpx, fpy = res
    
    v_idx = mesh.vert_df.index
    mesh.vert_df['f_line_x'] = flx[v_idx]
    mesh.vert_df['f_line_y'] = fly[v_idx]
    mesh.vert_df['f_area_x'] = fax[v_idx]
    mesh.vert_df['f_area_y'] = fay[v_idx]
    mesh.vert_df['f_perim_x'] = fpx[v_idx]
    mesh.vert_df['f_perim_y'] = fpy[v_idx]

    b_verts = mesh.get_boundary_vertices()
    if b_verts:
        for col in ['f_line_x', 'f_line_y', 'f_area_x', 'f_area_y', 'f_perim_x', 'f_perim_y']:
            mesh.vert_df.loc[b_verts, col] = 0.0

    mesh.vert_df['force_x'] = mesh.vert_df['f_line_x'] + mesh.vert_df['f_area_x'] + mesh.vert_df['f_perim_x']
    mesh.vert_df['force_y'] = mesh.vert_df['f_line_y'] + mesh.vert_df['f_area_y'] + mesh.vert_df['f_perim_y']
    
    return {"Total": E_area + E_perim + E_line, "Area": E_area, "Perimeter": E_perim, "Line": E_line}