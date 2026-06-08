import numpy as np

class FixedBoundaryTopology:
    def apply(self, mesh):
        mesh.use_pbc = False 
        mesh.pbc_v_pairs = []
        # Find raw open edges and freeze their boundary vertices mechanically
        for e_id in range(mesh.num_edges):
            if mesh.edge_twin[e_id] == -1:
                mesh.is_boundary_vert[mesh.edge_srce[e_id]] = True
                mesh.is_boundary_vert[mesh.edge_trgt[e_id]] = True
        return mesh


class PeriodicBoundaryTopology:
    def __init__(self, tolerance=1e-2):
        self.tolerance = tolerance

    def apply(self, mesh):
        """
        Directly links opposite boundaries together at the face-adjacency level.
        Uses a robust nearest-mirror projection strategy to handle staggered hex grids.
        Keeps perimeter coordinates mechanically immobile to avoid spatial tearing.
        """
        mesh.use_pbc = True
        mesh.is_boundary_vert.fill(False)
        mesh.pbc_v_pairs = [] 

        # 1. Gather all unassigned perimeter half-edges
        open_edges = np.where(mesh.edge_twin[:mesh.num_edges] == -1)[0]
        num_open = len(open_edges)
        
        if num_open == 0:
            return mesh

        # 2. Compute midpoints for all open edges
        midpoints = np.zeros((num_open, 2), dtype=np.float64)
        for idx, e_id in enumerate(open_edges):
            vs, vt = mesh.edge_srce[e_id], mesh.edge_trgt[e_id]
            midpoints[idx, 0] = (mesh.vert_x[vs] + mesh.vert_x[vt]) / 2.0
            midpoints[idx, 1] = (mesh.vert_y[vs] + mesh.vert_y[vt]) / 2.0

        Lx = mesh.exact_Lx
        Ly = mesh.exact_Ly

        # 3. Pair up opposite edges based on the nearest virtual mirror projection
        for i in range(num_open):
            e1 = open_edges[i]
            if mesh.edge_twin[e1] != -1:
                continue

            mx, my = midpoints[i, 0], midpoints[i, 1]

            # Generate all possible toroidal mirror reflections for this edge position
            mirrors = [
                (mx + Lx, my), (mx - Lx, my),  # X reflections
                (mx, my + Ly), (mx, my - Ly),  # Y reflections
                (mx + Lx, my + Ly), (mx - Lx, my - Ly), # Corner shifts
                (mx + Lx, my - Ly), (mx - Lx, my + Ly)
            ]

            best_j = -1
            min_dist = float('inf')

            # Find the opposite open edge that directly matches a mirror position
            for j in range(num_open):
                if i == j:
                    continue
                e2 = open_edges[j]
                if mesh.edge_twin[e2] != -1:
                    continue

                ox, oy = midpoints[j, 0], midpoints[j, 1]

                for target_x, target_y in mirrors:
                    dist = np.sqrt((ox - target_x)**2 + (oy - target_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_j = j

            # Link them if they are within geometric tolerance of the mirror projection
            if best_j != -1 and min_dist < 0.5:
                e2 = open_edges[best_j]
                mesh.edge_twin[e1] = e2
                mesh.edge_twin[e2] = e1

        # 4. Re-pin boundary vertices mechanically to enforce container immobility
        # Any edge that bridges across a macro mirror boundary is flagged to freeze its vertices.
        for e_id in range(mesh.num_edges):
            twin_id = mesh.edge_twin[e_id]
            if twin_id != -1:
                vs, vt = mesh.edge_srce[e_id], mesh.edge_trgt[e_id]
                vn_s = mesh.edge_srce[twin_id]
                
                # If the raw un-wrapped spatial distance is large, it's a periodic boundary edge
                raw_dx = abs(mesh.vert_x[vs] - mesh.vert_x[vn_s])
                raw_dy = abs(mesh.vert_y[vs] - mesh.vert_y[vn_s])
                
                if raw_dx > 0.5 * Lx or raw_dy > 0.5 * Ly:
                    mesh.is_boundary_vert[vs] = True
                    mesh.is_boundary_vert[vt] = True
            else:
                # Catch-all fallback for any remaining open segments
                mesh.is_boundary_vert[mesh.edge_srce[e_id]] = True
                mesh.is_boundary_vert[mesh.edge_trgt[e_id]] = True

        return mesh