import numpy as np
import pandas as pd
from core.mesh_engine import TissueMesh

def build_lattice(grid_type: str, nx: int, ny: int, noise_percent: float) -> TissueMesh:
    mesh = TissueMesh()
    
    # NORMALIZATION NETWORK:
    # If square, side length s = 1.0 results in Area = 1.0.
    # If hex, side length s ≈ 0.620403 results in Area = 1.0.
    if grid_type == 'square':
        s = 1.0
        h = 1.0
    elif grid_type == 'hex':
        s = np.sqrt(2.0 / (3.0 * np.sqrt(3.0))) 
        h = s * np.sqrt(3.0)
    else:
        s = 1.0
        h = 1.0

    unique_coords = set()
    for r in range(ny + (1 if grid_type == 'square' else 0)):
        for c in range(nx + (1 if grid_type == 'square' else 0)):
            if grid_type == 'hex':
                cx = c * s * 1.5; cy = r * h + (h / 2.0 if c % 2 == 1 else 0.0)
                coords = [(cx+s/2, cy+h/2), (cx-s/2, cy+h/2), (cx-s, cy), (cx-s/2, cy-h/2), (cx+s/2, cy-h/2), (cx+s, cy)]
            else:
                coords = [(float(c * s), float(r * s))]
            for x, y in coords: unique_coords.add((round(x, 4), round(y, 4)))

    sorted_coords = sorted(list(unique_coords), key=lambda p: (p[1], p[0]))
    coord_to_vid = {coord: i for i, coord in enumerate(sorted_coords)}
    
    mesh.num_verts = len(sorted_coords)
    for coord, vid in coord_to_vid.items():
        mesh.vert_x[vid], mesh.vert_y[vid] = coord[0], coord[1]

    valid_half_edges = []; face_vids = {}; f_id = 0
    for r in range(ny):
        for c in range(nx):
            if grid_type == 'hex':
                cx = c * s * 1.5; cy = r * h + (h / 2.0 if c % 2 == 1 else 0.0)
                raw_coords = [(cx+s/2, cy+h/2), (cx-s/2, cy+h/2), (cx-s, cy), (cx-s/2, cy-h/2), (cx+s/2, cy-h/2), (cx+s, cy)]
            else:
                raw_coords = [(c * s, r * s), ((c + 1) * s, r * s), ((c + 1) * s, (r + 1) * s), (c * s, (r + 1) * s)]
            cell_vids = [coord_to_vid[(round(x, 4), round(y, 4))] for x, y in raw_coords]
            face_vids[f_id] = cell_vids
            sides = len(cell_vids)
            for i in range(sides): valid_half_edges.append((cell_vids[i], cell_vids[(i + 1) % sides], f_id))
            f_id += 1

    mesh.num_faces = f_id
    mesh.face_target_area[:mesh.num_faces] = 1.0

    def he_midpoint(item):
        v1, v2, _ = item
        return (mesh.vert_y[v1] + mesh.vert_y[v2]) / 2.0, (mesh.vert_x[v1] + mesh.vert_x[v2]) / 2.0

    sorted_half_edges = sorted(valid_half_edges, key=he_midpoint)
    he_map = {(v1, v2): i for i, (v1, v2, _) in enumerate(sorted_half_edges)}
    mesh.num_edges = len(sorted_half_edges)

    for fid, cell_vids in face_vids.items():
        sides = len(cell_vids)
        cell_he_ids = [he_map[(cell_vids[i], cell_vids[(i + 1) % sides])] for i in range(sides)]
        for i, he_id in enumerate(cell_he_ids):
            mesh.edge_srce[he_id] = cell_vids[i]
            mesh.edge_trgt[he_id] = cell_vids[(i + 1) % sides]
            mesh.edge_face[he_id] = fid
            mesh.edge_next[he_id] = cell_he_ids[(i + 1) % sides]
            mesh.edge_prev[he_id] = cell_he_ids[(i - 1) % sides]
            mesh.edge_twin[he_id] = -1

    twin_map = {}
    for e_id in range(mesh.num_edges):
        key = (mesh.edge_srce[e_id], mesh.edge_trgt[e_id])
        rev_key = (mesh.edge_trgt[e_id], mesh.edge_srce[e_id])
        if rev_key in twin_map:
            twin_id = twin_map[rev_key]
            mesh.edge_twin[e_id] = twin_id
            mesh.edge_twin[twin_id] = e_id
        else: twin_map[key] = e_id

    for e_id in range(mesh.num_edges):
        if mesh.edge_twin[e_id] == -1:
            mesh.is_boundary_vert[mesh.edge_srce[e_id]] = True
            mesh.is_boundary_vert[mesh.edge_trgt[e_id]] = True

    if noise_percent > 0:
        # FIXED: max_shift is now multiplied by the baseline side length 's'
        # This keeps the scaling invariant across changes in geometry!
        max_shift = (noise_percent / 100.0) * (s / 2.0)
        for v in range(mesh.num_verts):
            if not mesh.is_boundary_vert[v]:
                mesh.vert_x[v] += np.random.uniform(-max_shift, max_shift)
                mesh.vert_y[v] += np.random.uniform(-max_shift, max_shift)
                
    return mesh