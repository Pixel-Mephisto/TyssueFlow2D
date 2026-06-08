import numpy as np
from model.physics import run_physics_step, resolve_t1_fast_numba
from model.biochemistry import run_biochem_step

def advance_simulation_frame(mesh, p_config, t1_config, bio_config, grid_type, global_dt, mech_substeps, chem_substeps):
    s_norm = 1.0 if grid_type == 'square' else 0.620403
    
    # 1. BIOCHEMICAL INTEGRATION (Stiff ODEs require smaller steps)
    # The chemistry advances by exactly global_dt minutes.
    dt_chem = global_dt / chem_substeps
    for _ in range(chem_substeps):
        run_biochem_step(mesh, bio_config, s_norm, dt_chem)

    # 2. MECHANICAL INTEGRATION (Physical flow requires topology checks)
    # The physics advances by exactly global_dt minutes.
    dt_mech = global_dt / mech_substeps
    energy = 0.0
    total_swaps = 0
    
    for _ in range(mech_substeps):
        energy = run_physics_step(mesh, p_config)
        
        for v in range(mesh.num_verts):
            if not mesh.is_boundary_vert[v]:
                disp_x = dt_mech * mesh.force_x[v]
                disp_y = dt_mech * mesh.force_y[v]
                if p_config.use_brownian:
                    disp_x += np.random.normal(0, p_config.brownian_mag) * np.sqrt(dt_mech)
                    disp_y += np.random.normal(0, p_config.brownian_mag) * np.sqrt(dt_mech)
                mesh.vert_x[v] += disp_x
                mesh.vert_y[v] += disp_y
                
        # Topology must be checked after every mechanical micro-step to prevent mesh inversion
        swaps = resolve_t1_fast_numba(
            mesh.num_edges, mesh.vert_x, mesh.vert_y,
            mesh.edge_srce, mesh.edge_trgt, mesh.edge_twin,
            mesh.edge_next, mesh.edge_prev, mesh.edge_face,
            t1_config.threshold, t1_config.rest_length
        )
        total_swaps += swaps
        
    return total_swaps, energy