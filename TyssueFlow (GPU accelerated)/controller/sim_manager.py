import numpy as np
from model.gpu_physics import compute_mechanics_and_scan_gpu
from model.physics import execute_targeted_t1_swaps
from model.biochemistry import run_biochem_step

def advance_simulation_frame(mesh, p_config, t1_config, bio_config, grid_type, global_dt, mech_substeps, chem_substeps, frames_to_skip):
    s_norm = 1.0 if grid_type == 'square' else 0.620403
    total_swaps = 0
    
    for _ in range(frames_to_skip):
        
        # 1. BIOLOGICAL INTEGRATION
        dt_chem = global_dt / chem_substeps
        for _ in range(chem_substeps):
            run_biochem_step(mesh, bio_config, s_norm, dt_chem)

        # 2. MECHANICAL INTEGRATION
        dt_mech = global_dt / mech_substeps
        for _ in range(mech_substeps):
            
            flagged_short_edges = compute_mechanics_and_scan_gpu(mesh, p_config, t1_config)
            
            for v in range(mesh.num_verts):
                if not mesh.is_boundary_vert[v]:
                    disp_x = dt_mech * mesh.force_x[v]
                    disp_y = dt_mech * mesh.force_y[v]
                    if p_config.use_brownian:
                        disp_x += np.random.normal(0, p_config.brownian_mag) * np.sqrt(dt_mech)
                        disp_y += np.random.normal(0, p_config.brownian_mag) * np.sqrt(dt_mech)
                    mesh.vert_x[v] += disp_x
                    mesh.vert_y[v] += disp_y
                    
            if len(flagged_short_edges) > 0:
                swaps = execute_targeted_t1_swaps(
                    flagged_short_edges, mesh.vert_x, mesh.vert_y,
                    mesh.edge_srce, mesh.edge_trgt, mesh.edge_twin,
                    mesh.edge_next, mesh.edge_prev, mesh.edge_face,
                    t1_config.rest_length
                )
                total_swaps += swaps
                
    return total_swaps