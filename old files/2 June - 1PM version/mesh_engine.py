import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

class TissueMesh:
    """
    Research-Grade Vertex Model Data Engine.
    Maintains rigorous Half-Edge topology with strictly continuous indexing.
    """
    def __init__(self) -> None:
        self.vert_df = pd.DataFrame(columns=['x', 'y', 'force_x', 'force_y'])
        self.vert_df.index.name = 'vert_id'
        self.face_df = pd.DataFrame(columns=['target_area', 'notch', 'delta'])
        self.face_df.index.name = 'face_id'
        self.edge_df = pd.DataFrame(columns=['srce', 'trgt', 'face', 'next', 'prev', 'tension'])
        self.edge_df.index.name = 'edge_id'

    def generate_mesh(self, grid_type: str, nx: int, ny: int) -> None:
        """Universal factory interface for lattice generation."""
        if grid_type.lower() == 'square':
            self._generate_sweep_sorted_grid(nx, ny, 'square')
        elif grid_type.lower() == 'hex':
            self._generate_sweep_sorted_grid(nx, ny, 'hex')
        else:
            raise ValueError("Unsupported grid type. Must be 'square' or 'hex'.")

    def _generate_sweep_sorted_grid(self, nx: int, ny: int, grid_type: str) -> None:
        """Generates grid topology with strict sweep-line sorting and CCW winding."""
        s = 1.0
        h = s * np.sqrt(3) if grid_type == 'hex' else s

        # --- PASS 1: Spatial Discovery ---
        unique_coords = set()
        for r in range(ny + (1 if grid_type == 'square' else 0)):
            for c in range(nx + (1 if grid_type == 'square' else 0)):
                if grid_type == 'hex':
                    cx = c * s * 1.5
                    cy = r * h + (h / 2.0 if c % 2 == 1 else 0.0)
                    # Strict CCW Winding: TR, TL, L, BL, BR, R
                    coords = [
                        (cx + s/2, cy + h/2), (cx - s/2, cy + h/2), (cx - s, cy),
                        (cx - s/2, cy - h/2), (cx + s/2, cy - h/2), (cx + s, cy)
                    ]
                else:
                    coords = [(float(c * s), float(r * s))]
                
                for x, y in coords:
                    unique_coords.add((round(x, 4), round(y, 4)))

        # --- PASS 2: Deterministic Vertex Mapping ---
        sorted_coords = sorted(list(unique_coords), key=lambda p: (p[1], p[0]))
        coord_to_vid = {coord: i + 1 for i, coord in enumerate(sorted_coords)}

        verts = {vid: {'x': c[0], 'y': c[1], 'force_x': 0.0, 'force_y': 0.0} 
                 for c, vid in coord_to_vid.items()}

        # --- PASS 3: Face & Edge Discovery ---
        valid_half_edges = []
        face_vids = {}
        f_id = 1
        
        for r in range(ny):
            for c in range(nx):
                if grid_type == 'hex':
                    cx = c * s * 1.5
                    cy = r * h + (h / 2.0 if c % 2 == 1 else 0.0)
                    raw_coords = [
                        (cx + s/2, cy + h/2), (cx - s/2, cy + h/2), (cx - s, cy),
                        (cx - s/2, cy - h/2), (cx + s/2, cy - h/2), (cx + s, cy)
                    ]
                else:
                    raw_coords = [
                        (c * s, r * s), ((c + 1) * s, r * s),
                        ((c + 1) * s, (r + 1) * s), (c * s, (r + 1) * s)
                    ]
                    
                cell_vids = [coord_to_vid[(round(x, 4), round(y, 4))] for x, y in raw_coords]
                face_vids[f_id] = cell_vids

                sides = len(cell_vids)
                for i in range(sides):
                    valid_half_edges.append((cell_vids[i], cell_vids[(i + 1) % sides], f_id))
                f_id += 1

        # --- PASS 4: Sweep-Line Edge Sorting ---
        def he_midpoint(item: Tuple[int, int, int]) -> Tuple[float, float]:
            v1, v2, _ = item
            return (verts[v1]['y'] + verts[v2]['y']) / 2.0, (verts[v1]['x'] + verts[v2]['x']) / 2.0

        sorted_half_edges = sorted(valid_half_edges, key=he_midpoint)
        he_map = {(v1, v2): i + 1 for i, (v1, v2, fid) in enumerate(sorted_half_edges)}

        # --- PASS 5: DataFrame Compilation & Pointer Linking ---
        faces, edges = {}, {}
        for fid, cell_vids in face_vids.items():
            faces[fid] = {'target_area': 1.0, 'notch': np.random.uniform(0, 1), 'delta': np.random.uniform(0, 1)}

            sides = len(cell_vids)
            cell_he_ids = [he_map[(cell_vids[i], cell_vids[(i + 1) % sides])] for i in range(sides)]

            for i, he_id in enumerate(cell_he_ids):
                edges[he_id] = {
                    'srce': cell_vids[i], 
                    'trgt': cell_vids[(i + 1) % sides], 
                    'face': fid, 
                    'next': cell_he_ids[(i + 1) % sides], 
                    'prev': cell_he_ids[(i - 1) % sides], 
                    'tension': 1.0
                }

        self.vert_df = pd.DataFrame.from_dict(verts, orient='index')
        self.face_df = pd.DataFrame.from_dict(faces, orient='index')
        self.edge_df = pd.DataFrame.from_dict(edges, orient='index').astype(int)

    def apply_vertex_noise(self, noise_percent: float) -> None:
        """
        Applies topological noise to interior vertices using fast vectorized NumPy arrays.
        """
        if self.edge_df.empty: return

        # 1. Vectorized Boundary Detection
        pairs = np.sort(self.edge_df[['srce', 'trgt']].values, axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        boundary_edges = unique_pairs[counts == 1]
        boundary_vids = np.unique(boundary_edges)

        # 2. Boolean Masking for Interior Vertices
        interior_mask = ~self.vert_df.index.isin(boundary_vids)
        
        # 3. Vectorized Noise Application
        max_shift = (noise_percent / 100.0) / 2.0
        n_interior = interior_mask.sum()
        
        if n_interior > 0:
            shifts = np.random.uniform(-max_shift, max_shift, size=(n_interior, 2))
            self.vert_df.loc[interior_mask, ['x', 'y']] += shifts

    # =================================================================
    # NEW PHYSICS EXECUTORS: APPENED TO YOUR EXACT GEOMETRY ENGINE
    # =================================================================
    
    def compute_geometry(self) -> None:
        """Calculates areas and perimeters needed for the Hamiltonian forces."""
        if self.face_df.empty: return
        
        # Ensure calculation columns exist without breaking your initial Face setup
        if 'calc_area' not in self.face_df.columns:
            self.face_df['calc_area'] = 0.0
            self.face_df['calc_perimeter'] = 0.0
            
        for face_id in self.face_df.index:
            if face_id == -1: continue # Ghost cell protection
            
            face_edges = self.edge_df[self.edge_df['face'] == face_id]
            if face_edges.empty: continue
            
            curr_edge = start_edge = face_edges.index[0]
            ordered_verts = []
            perimeter = 0.0
            
            while True:
                row = self.edge_df.loc[curr_edge]
                v1, v2 = self.vert_df.loc[row['srce']], self.vert_df.loc[row['trgt']]
                
                ordered_verts.append((v1['x'], v1['y']))
                perimeter += np.hypot(v2['x'] - v1['x'], v2['y'] - v1['y'])
                
                curr_edge = row['next']
                if curr_edge == start_edge: break
                
            x = [p[0] for p in ordered_verts]
            y = [p[1] for p in ordered_verts]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            self.face_df.at[face_id, 'calc_area'] = area
            self.face_df.at[face_id, 'calc_perimeter'] = perimeter

    def get_boundary_vertices(self) -> list:
        """Leverages your vectorized noise boundary logic to find outer vertices."""
        if self.edge_df.empty: return []
        pairs = np.sort(self.edge_df[['srce', 'trgt']].values, axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        boundary_edges = unique_pairs[counts == 1]
        return np.unique(boundary_edges).tolist()

    def step_viscous(self, dt: float = 0.02, eta: float = 1.0) -> None:
        """Applies overdamped dynamics with STRICT absolute boundary immobilization."""
        displacement_x = (dt / eta) * self.vert_df['force_x'].copy()
        displacement_y = (dt / eta) * self.vert_df['force_y'].copy()
        
        # Absolute structural lockdown of boundaries
        b_verts = self.get_boundary_vertices()
        if b_verts:
            displacement_x.loc[b_verts] = 0.0
            displacement_y.loc[b_verts] = 0.0
        
        self.vert_df['x'] += displacement_x
        self.vert_df['y'] += displacement_y
        
        self.compute_geometry()