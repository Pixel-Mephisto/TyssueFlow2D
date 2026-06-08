import numpy as np
from numba import njit, prange

@njit
def _compute_trans_signals(num_edges, edge_srce, edge_trgt, edge_face, edge_twin, vert_x, vert_y, face_N, face_D, s_norm):
    num_faces = len(face_N)
    sD = np.zeros(num_faces, dtype=np.float64)
    sN = np.zeros(num_faces, dtype=np.float64)
    
    for e in range(num_edges):
        twin = edge_twin[e]
        if twin != -1:
            f_self = edge_face[e]
            f_neighbor = edge_face[twin]
            
            if f_self != -1 and f_neighbor != -1:
                v1, v2 = edge_srce[e], edge_trgt[e]
                dx, dy = vert_x[v2] - vert_x[v1], vert_y[v2] - vert_y[v1]
                length = np.sqrt(dx*dx + dy*dy)
                contact_weight = length / s_norm
                
                sD[f_self] += face_D[f_neighbor] * contact_weight
                sN[f_self] += face_N[f_neighbor] * contact_weight
                
    return sN, sD

@njit(parallel=True)
def _integrate_odes(num_faces, face_N, face_D, face_I, sN, sD, gN, gD, kC, gamma, gammaI, I0, kT0, dt_chem):
    for f in prange(num_faces):
        if f == -1: continue
        N, D, I = face_N[f], face_D[f], face_I[f]
        
        I0_sq = I0 * I0
        I_sq = I * I
        hN = (I0_sq + 2.0 * I_sq) / (I0_sq + I_sq) 
        hD = I0_sq / (I0_sq + I_sq)                 
        
        dN = (gN * hN - kC * N * D - kT0 * N * sD[f] - gamma * N) * dt_chem
        dD = (gD * hD - kC * N * D - kT0 * D * sN[f] - gamma * D) * dt_chem
        dI = (kT0 * N * sD[f] - gammaI * I) * dt_chem
        
        face_N[f] = max(0.0, N + dN)
        face_D[f] = max(0.0, D + dD)
        face_I[f] = max(0.0, I + dI)

def run_biochem_step(mesh, bio_config, s_norm, dt_chem):
    sN, sD = _compute_trans_signals(
        mesh.num_edges, mesh.edge_srce, mesh.edge_trgt, mesh.edge_face, mesh.edge_twin,
        mesh.vert_x, mesh.vert_y, mesh.face_N, mesh.face_D, s_norm
    )
    _integrate_odes(
        mesh.num_faces, mesh.face_N, mesh.face_D, mesh.face_I, sN, sD,
        bio_config.gN, bio_config.gD, bio_config.kC, bio_config.gamma, bio_config.gammaI, 
        bio_config.I0, bio_config.kT0, dt_chem
    )