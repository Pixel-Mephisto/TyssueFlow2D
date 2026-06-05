import numpy as np
from typing import Dict
from mesh_engine import TissueMesh

def compute_system_energy(mesh: TissueMesh, 
                          K_A: float = 1.0, 
                          Gamma_P: float = 0.1, 
                          p_0: float = 3.81,
                          Lambda: float = 1.0) -> Dict[str, float]:
    """Computes Hamiltonian energy and isolated analytical force gradients."""
    mesh.compute_geometry()
    
    # 1. Initialize dedicated isolated force columns
    for col in ['f_line_x', 'f_line_y', 'f_area_x', 'f_area_y', 'f_perim_x', 'f_perim_y']:
        mesh.vert_df[col] = 0.0
        
    # 2. Modular Physics Calls
    E_line = _apply_line_tension(mesh, Lambda)
    E_area, E_perim = _apply_face_mechanics(mesh, K_A, Gamma_P, p_0)
    
    # 3. Secure Boundaries & Calculate Net Force
    _freeze_boundaries(mesh)
    
    # Sum the isolated forces to get the total Net Force
    mesh.vert_df['force_x'] = mesh.vert_df['f_line_x'] + mesh.vert_df['f_area_x'] + mesh.vert_df['f_perim_x']
    mesh.vert_df['force_y'] = mesh.vert_df['f_line_y'] + mesh.vert_df['f_area_y'] + mesh.vert_df['f_perim_y']
    
    total_energy = E_area + E_perim + E_line
    return {"Total": total_energy, "Area": E_area, "Perimeter": E_perim, "Line": E_line}

def _apply_line_tension(mesh: TissueMesh, Lambda: float) -> float:
    E_line = 0.0
    for edge_id, row in mesh.edge_df.iterrows():
        v1_id, v2_id = row['srce'], row['trgt']
        x1, y1 = mesh.vert_df.at[v1_id, 'x'], mesh.vert_df.at[v1_id, 'y']
        x2, y2 = mesh.vert_df.at[v2_id, 'x'], mesh.vert_df.at[v2_id, 'y']
        
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-6: continue
        
        E_line += 0.5 * Lambda * length
        
        f_x, f_y = 0.5 * Lambda * (dx / length), 0.5 * Lambda * (dy / length)
        
        mesh.vert_df.at[v1_id, 'f_line_x'] += f_x
        mesh.vert_df.at[v1_id, 'f_line_y'] += f_y
        mesh.vert_df.at[v2_id, 'f_line_x'] -= f_x
        mesh.vert_df.at[v2_id, 'f_line_y'] -= f_y
    return E_line

def _apply_face_mechanics(mesh: TissueMesh, K_A: float, Gamma_P: float, p_0: float):
    E_area, E_perim = 0.0, 0.0
    for face_id, row in mesh.face_df.iterrows():
        A_c, P_c, A_0 = row['calc_area'], row['calc_perimeter'], row['target_area']
        P_0 = p_0 * np.sqrt(A_0)
        
        pressure = -K_A * (A_c - A_0)
        contractility = -Gamma_P * (P_c - P_0)
        
        E_area += 0.5 * K_A * (A_c - A_0)**2
        E_perim += 0.5 * Gamma_P * (P_c - P_0)**2
        
        face_edges = mesh.edge_df[mesh.edge_df['face'] == face_id]
        if face_edges.empty: continue
        
        curr_edge = start_edge = face_edges.index[0]
        ordered_edges = []
        while True:
            ordered_edges.append(curr_edge)
            curr_edge = mesh.edge_df.loc[curr_edge, 'next']
            if curr_edge == start_edge: break
            
        n = len(ordered_edges)
        for i in range(n):
            e_curr, e_prev = ordered_edges[i], ordered_edges[(i - 1) % n]
            v_i = mesh.edge_df.at[e_curr, 'srce']
            v_next = mesh.edge_df.at[e_curr, 'trgt']
            v_prev = mesh.edge_df.at[e_prev, 'srce']
            
            xi, yi = mesh.vert_df.at[v_i, 'x'], mesh.vert_df.at[v_i, 'y']
            x_next, y_next = mesh.vert_df.at[v_next, 'x'], mesh.vert_df.at[v_next, 'y']
            x_prev, y_prev = mesh.vert_df.at[v_prev, 'x'], mesh.vert_df.at[v_prev, 'y']
            
            dx1, dy1 = xi - x_prev, yi - y_prev
            dx2, dy2 = x_next - xi, y_next - yi
            len1, len2 = np.hypot(dx1, dy1), np.hypot(dx2, dy2)
            
            if len1 > 1e-6 and len2 > 1e-6:
                mesh.vert_df.at[v_i, 'f_perim_x'] += contractility * ((dx1 / len1) - (dx2 / len2))
                mesh.vert_df.at[v_i, 'f_perim_y'] += contractility * ((dy1 / len1) - (dy2 / len2))
                
            mesh.vert_df.at[v_i, 'f_area_x'] += pressure * (0.5 * (y_next - y_prev))
            mesh.vert_df.at[v_i, 'f_area_y'] += pressure * (0.5 * (x_prev - x_next))
            
    return E_area, E_perim

def _freeze_boundaries(mesh: TissueMesh):
    all_edges = set(mesh.edge_df.index)
    internal_edges = set(mesh.edge_df.dropna(subset=['face']).index)
    boundary_edges = all_edges - internal_edges
    
    boundary_verts = set()
    for e_id in boundary_edges:
        boundary_verts.add(mesh.edge_df.at[e_id, 'srce'])
        boundary_verts.add(mesh.edge_df.at[e_id, 'trgt'])
        
    if boundary_verts:
        b_list = list(boundary_verts)
        for col in ['f_line_x', 'f_line_y', 'f_area_x', 'f_area_y', 'f_perim_x', 'f_perim_y']:
            mesh.vert_df.loc[b_list, col] = 0.0