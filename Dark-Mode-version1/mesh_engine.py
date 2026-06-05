import pandas as pd
import numpy as np
from typing import Tuple, Optional

class TissueMesh:
    """
    Research-Grade Vertex Model Data Engine.
    Maintains rigorous Half-Edge topology with continuous indexing and T1 capabilities.
    """
    def __init__(self) -> None:
        self.vert_df = pd.DataFrame(columns=['x', 'y', 'force_x', 'force_y'])
        self.vert_df.index.name = 'vert_id'
        
        self.face_df = pd.DataFrame(columns=['target_area', 'notch', 'delta', 'calc_area', 'calc_perimeter'])
        self.face_df.index.name = 'face_id'
        
        self.edge_df = pd.DataFrame(columns=['srce', 'trgt', 'face', 'next', 'prev', 'twin', 'tension'])
        self.edge_df.index.name = 'edge_id'

    def generate_mesh(self, grid_type: str, nx: int, ny: int) -> None:
        """Universal factory interface for lattice generation."""
        s = 1.0
        h = s * np.sqrt(3) if grid_type == 'hex' else s

        # --- PASS 1: Spatial Discovery ---
        unique_coords = set()
        for r in range(ny + (1 if grid_type == 'square' else 0)):
            for c in range(nx + (1 if grid_type == 'square' else 0)):
                if grid_type == 'hex':
                    cx = c * s * 1.5
                    cy = r * h + (h / 2.0 if c % 2 == 1 else 0.0)
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
        verts = {vid: {'x': c[0], 'y': c[1], 'force_x': 0.0, 'force_y': 0.0} for c, vid in coord_to_vid.items()}

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
            faces[fid] = {'target_area': 1.0, 'notch': np.random.uniform(0, 1), 'delta': np.random.uniform(0, 1), 'calc_area': 0.0, 'calc_perimeter': 0.0}
            sides = len(cell_vids)
            cell_he_ids = [he_map[(cell_vids[i], cell_vids[(i + 1) % sides])] for i in range(sides)]
            for i, he_id in enumerate(cell_he_ids):
                edges[he_id] = {
                    'srce': cell_vids[i], 'trgt': cell_vids[(i + 1) % sides], 
                    'face': fid, 'next': cell_he_ids[(i + 1) % sides], 
                    'prev': cell_he_ids[(i - 1) % sides], 
                    'twin': -1, 'tension': 1.0
                }

        self.vert_df = pd.DataFrame.from_dict(verts, orient='index')
        self.face_df = pd.DataFrame.from_dict(faces, orient='index')
        self.edge_df = pd.DataFrame.from_dict(edges, orient='index').astype(int)

        # --- PASS 6: Cross-Table Twin Synchronization ---
        twin_map = {}
        for e_id, row in self.edge_df.iterrows():
            key = (row['srce'], row['trgt'])
            rev_key = (row['trgt'], row['srce'])
            if rev_key in twin_map:
                twin_id = twin_map[rev_key]
                self.edge_df.at[e_id, 'twin'] = twin_id
                self.edge_df.at[twin_id, 'twin'] = e_id
            else:
                twin_map[key] = e_id

    def apply_vertex_noise(self, noise_percent: float) -> None:
        """Applies random spatial perturbation strictly to internal vertices."""
        if self.edge_df.empty: return
        pairs = np.sort(self.edge_df[['srce', 'trgt']].values, axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        boundary_edges = unique_pairs[counts == 1]
        boundary_vids = np.unique(boundary_edges)

        interior_mask = ~self.vert_df.index.isin(boundary_vids)
        max_shift = (noise_percent / 100.0) / 2.0
        n_interior = interior_mask.sum()
        
        if n_interior > 0:
            shifts = np.random.uniform(-max_shift, max_shift, size=(n_interior, 2))
            self.vert_df.loc[interior_mask, ['x', 'y']] += shifts

    def compute_geometry(self) -> None:
        """Calculates explicit area and perimeter invariants for all functional faces."""
        if self.face_df.empty: return
        for face_id in self.face_df.index:
            if face_id == -1: continue 
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
        """Identifies open-boundary topological indices from non-twinned half-edges."""
        if self.edge_df.empty: return []
        pairs = np.sort(self.edge_df[['srce', 'trgt']].values, axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        boundary_edges = unique_pairs[counts == 1]
        return np.unique(boundary_edges).tolist()

    def step_viscous(self, dt: float = 0.02, eta: float = 1.0, brownian_noise: float = 0.0) -> None:
        """Applies overdamped dynamics coupled with stochastic Langevin thermal vectors."""
        displacement_x = (dt / eta) * self.vert_df['force_x'].copy()
        displacement_y = (dt / eta) * self.vert_df['force_y'].copy()
        
        # Apply Brownian motion mapped correctly with sqrt(dt)
        if brownian_noise > 0.0:
            num_verts = len(self.vert_df)
            noise_x = np.random.normal(0, brownian_noise, num_verts) * np.sqrt(dt)
            noise_y = np.random.normal(0, brownian_noise, num_verts) * np.sqrt(dt)
            displacement_x += noise_x
            displacement_y += noise_y
        
        # Rigorous boundary immobilization lockdown
        b_verts = self.get_boundary_vertices()
        if b_verts:
            displacement_x.loc[b_verts] = 0.0
            displacement_y.loc[b_verts] = 0.0
            
        self.vert_df['x'] += displacement_x
        self.vert_df['y'] += displacement_y
        self.compute_geometry()

    # =========================================================================
    # TOPOLOGICAL SURGERY & DYNAMIC RESOLVER METHODS
    # =========================================================================
    
    def perform_t1_swap(self, he1_id: int, rest_length: float = 0.08) -> Tuple[bool, Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Executes a mathematically clean CCW topological neighborhood rewiring pass."""
        he1 = he1_id
        he2 = int(self.edge_df.at[he1, 'twin'])
        
        if pd.isna(he2) or he2 == -1: 
            return False, None, None, None, None

        v1 = int(self.edge_df.at[he1, 'srce'])
        v2 = int(self.edge_df.at[he1, 'trgt'])

        he1_prev = int(self.edge_df.at[he1, 'prev'])
        he1_next = int(self.edge_df.at[he1, 'next'])
        he2_prev = int(self.edge_df.at[he2, 'prev'])
        he2_next = int(self.edge_df.at[he2, 'next'])

        he1_prev_twin = int(self.edge_df.at[he1_prev, 'twin'])
        he1_next_twin = int(self.edge_df.at[he1_next, 'twin'])
        he2_prev_twin = int(self.edge_df.at[he2_prev, 'twin'])
        he2_next_twin = int(self.edge_df.at[he2_next, 'twin'])

        # Protect against open-boundary void lookup errors
        if -1 in [he1_prev_twin, he1_next_twin, he2_prev_twin, he2_next_twin]:
            return False, None, None, None, None

        f1 = self.edge_df.at[he1, 'face']
        f2 = self.edge_df.at[he2, 'face']
        f3 = self.edge_df.at[he1_prev_twin, 'face'] 
        f4 = self.edge_df.at[he1_next_twin, 'face'] 

        # --- GEOMETRIC AXIS FLIP ---
        x1, y1 = self.vert_df.at[v1, 'x'], self.vert_df.at[v1, 'y']
        x2, y2 = self.vert_df.at[v2, 'x'], self.vert_df.at[v2, 'y']
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-6: 
            return False, None, None, None, None
        
        nx, ny = -dy / length, dx / length 
        
        self.vert_df.at[v1, 'x'], self.vert_df.at[v1, 'y'] = cx + nx * rest_length / 2.0, cy + ny * rest_length / 2.0
        self.vert_df.at[v2, 'x'], self.vert_df.at[v2, 'y'] = cx - nx * rest_length / 2.0, cy - ny * rest_length / 2.0

        # --- TOPOLOGICAL REWIRING POINTER SURGERY ---
        self.edge_df.at[he1_next, 'srce'] = v1
        self.edge_df.at[he1_next_twin, 'trgt'] = v1
        
        self.edge_df.at[he2_next, 'srce'] = v2
        self.edge_df.at[he2_next_twin, 'trgt'] = v2

        self.edge_df.at[he1_prev, 'next'] = he1_next
        self.edge_df.at[he1_next, 'prev'] = he1_prev

        self.edge_df.at[he2_prev, 'next'] = he2_next
        self.edge_df.at[he2_next, 'prev'] = he2_prev

        self.edge_df.at[he1, 'face'] = f4
        self.edge_df.at[he1_next_twin, 'next'] = he1
        self.edge_df.at[he1, 'prev'] = he1_next_twin
        self.edge_df.at[he1, 'next'] = he2_prev_twin
        self.edge_df.at[he2_prev_twin, 'prev'] = he1

        self.edge_df.at[he2, 'face'] = f3
        self.edge_df.at[he2_next_twin, 'next'] = he2
        self.edge_df.at[he2, 'prev'] = he2_next_twin
        self.edge_df.at[he2, 'next'] = he1_prev_twin
        self.edge_df.at[he1_prev_twin, 'prev'] = he2
        
        return True, f1, f2, f3, f4

    def resolve_t1_transitions(self, threshold: float = 0.05, rest_length: float = 0.08, max_iter: int = 5) -> int:
        """Iteratively scans and resolves compressed edges below critical thresholds."""
        total_swaps = 0
        for _ in range(max_iter):
            self.compute_geometry()
            short_edge_id = None
            
            for e_id, row in self.edge_df[self.edge_df['twin'] != -1].iterrows():
                v1, v2 = row['srce'], row['trgt']
                x1, y1 = self.vert_df.loc[v1, ['x', 'y']]
                x2, y2 = self.vert_df.loc[v2, ['x', 'y']]
                length = np.hypot(x2 - x1, y2 - y1)
                
                if length < threshold:
                    he1, he2 = e_id, row['twin']
                    n1, n2 = self.edge_df.at[he1, 'prev'], self.edge_df.at[he1, 'next']
                    n3, n4 = self.edge_df.at[he2, 'prev'], self.edge_df.at[he2, 'next']
                    
                    twins = [
                        self.edge_df.at[n1, 'twin'], self.edge_df.at[n2, 'twin'],
                        self.edge_df.at[n3, 'twin'], self.edge_df.at[n4, 'twin']
                    ]
                    
                    if -1 not in twins:
                        short_edge_id = e_id
                        break 
            
            if short_edge_id is None:
                break 
                
            success, f1, f2, f3, f4 = self.perform_t1_swap(short_edge_id, rest_length=rest_length)
            if success:
                total_swaps += 1
                
        if total_swaps > 0:
            self.compute_geometry()
            
        return total_swaps