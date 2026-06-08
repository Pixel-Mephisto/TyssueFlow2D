import numpy as np
from numba import njit, prange

@njit
def execute_targeted_t1_swaps(short_edges, vert_x, vert_y, edge_srce, edge_trgt, edge_twin, edge_next, edge_prev, edge_face, rest_length):
    """Executes surgery strictly on edges flagged by the GPU with Zero-Division safeguards."""
    swaps = 0
    for i in range(len(short_edges)):
        e = short_edges[i]
        twin = edge_twin[e]
        if twin == -1: continue

        v1, v2 = edge_srce[e], edge_trgt[e]
        
        dx, dy = vert_x[v2] - vert_x[v1], vert_y[v2] - vert_y[v1]
        length = np.sqrt(dx*dx + dy*dy)
        
        # PREVENT ZERO-DIVISION CRASH
        if length < 1e-8: 
            continue
            
        he1_prev, he1_next = edge_prev[e], edge_next[e]
        he2_prev, he2_next = edge_prev[twin], edge_next[twin]
        
        he1_prev_twin = edge_twin[he1_prev]
        he1_next_twin = edge_twin[he1_next]
        he2_prev_twin = edge_twin[he2_prev]
        he2_next_twin = edge_twin[he2_next]
        
        if -1 in [he1_prev_twin, he1_next_twin, he2_prev_twin, he2_next_twin]: continue
            
        f3 = edge_face[he1_prev_twin]
        f4 = edge_face[he1_next_twin]
        
        cx, cy = (vert_x[v1] + vert_x[v2]) / 2.0, (vert_y[v1] + vert_y[v2]) / 2.0
        nx, ny = -dy / length, dx / length
        
        vert_x[v1], vert_y[v1] = cx + nx * rest_length / 2.0, cy + ny * rest_length / 2.0
        vert_x[v2], vert_y[v2] = cx - nx * rest_length / 2.0, cy - ny * rest_length / 2.0
        
        edge_srce[he1_next] = v1; edge_trgt[he1_next_twin] = v1
        edge_srce[he2_next] = v2; edge_trgt[he2_next_twin] = v2
        
        edge_next[he1_prev] = he1_next; edge_prev[he1_next] = he1_prev
        edge_next[he2_prev] = he2_next; edge_prev[he2_next] = he2_prev
        
        edge_face[e] = f4
        edge_next[he1_next_twin] = e; edge_prev[e] = he1_next_twin
        edge_next[e] = he2_prev_twin; edge_prev[he2_prev_twin] = e
        
        edge_face[twin] = f3
        edge_next[he2_next_twin] = twin; edge_prev[twin] = he2_next_twin
        edge_next[twin] = he1_prev_twin; edge_prev[he1_prev_twin] = twin
        
        swaps += 1
            
    return swaps