import numpy as np
from model.physics import run_physics_step, resolve_t1_fast_numba

def advance_simulation_frame(mesh, p_config, t1_config):
    # Retrieve the dynamically computed physical energy!
    energy = run_physics_step(mesh, p_config)
    dt = p_config.dt
    
    for v in range(mesh.num_verts):
        if not mesh.is_boundary_vert[v]:
            disp_x = dt * mesh.force_x[v]
            disp_y = dt * mesh.force_y[v]
            if p_config.use_brownian:
                disp_x += np.random.normal(0, p_config.brownian_mag) * np.sqrt(dt)
                disp_y += np.random.normal(0, p_config.brownian_mag) * np.sqrt(dt)
            mesh.vert_x[v] += disp_x
            mesh.vert_y[v] += disp_y
            
    swaps = resolve_t1_fast_numba(
        mesh.num_edges, mesh.vert_x, mesh.vert_y,
        mesh.edge_srce, mesh.edge_trgt, mesh.edge_twin,
        mesh.edge_next, mesh.edge_prev, mesh.edge_face,
        t1_config.threshold, t1_config.rest_length
    )
    return swaps, energy