import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mesh_engine import TissueMesh

def plot_mesh(mesh: TissueMesh, 
              ax=None,
              show_faces: bool = True, 
              show_edges: bool = True, 
              show_verts: bool = True, 
              show_labels: bool = True,
              show_f_line: bool = False,
              show_f_area: bool = False,
              show_f_perim: bool = False,
              show_f_total: bool = False,
              title: str = "Tissue Topology") -> None:
    
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone = True
        
    ax.clear()
    ax.set_facecolor('#f8fafc')
    
    num_edges = len(mesh.edge_df)
    render_text = show_labels and num_edges < 200 
    
    # 1. RENDER FACES
    for face_id in mesh.face_df.index:
        if face_id == -1: continue # Ghost cell protection
        
        face_edges = mesh.edge_df[mesh.edge_df['face'] == face_id]
        if face_edges.empty: continue
        
        curr_edge = start_edge = face_edges.index[0]
        poly_coords = []
        visited = set()
        
        while curr_edge not in visited:
            visited.add(curr_edge)
            srce_vert = mesh.vert_df.loc[mesh.edge_df.loc[curr_edge, 'srce']]
            poly_coords.append([srce_vert['x'], srce_vert['y']])
            curr_edge = mesh.edge_df.loc[curr_edge, 'next']
            if curr_edge == start_edge: break
            
        if len(poly_coords) > 2:
            poly = patches.Polygon(poly_coords, closed=True, fill=show_faces, 
                                   facecolor='#e2e8f0', edgecolor='#64748b', lw=0.5, alpha=0.8)
            ax.add_patch(poly)
            if render_text:
                cx, cy = np.mean(poly_coords, axis=0)
                ax.text(cx, cy, f"F{face_id}", color='#1e293b', fontsize=8, ha='center', va='center')

    # 2. RENDER EDGES & VERTS
    if show_edges:
        for edge_id, edge_row in mesh.edge_df.iterrows():
            v1, v2 = mesh.vert_df.loc[edge_row['srce']], mesh.vert_df.loc[edge_row['trgt']]
            ax.plot([v1['x'], v2['x']], [v1['y'], v2['y']], color='#94a3b8', lw=1.0, zorder=1)

    if show_verts:
        ax.scatter(mesh.vert_df['x'], mesh.vert_df['y'], color='#334155', s=15, zorder=3)

    # --- 3. VISUAL MASKING: HIDE BOUNDARY FORCES ---
    # Fetch boundary vertices and create an interior-only mask
    b_verts = mesh.get_boundary_vertices()
    interior_mask = ~mesh.vert_df.index.isin(b_verts)
    
    # Extract ONLY interior coordinates for the arrows
    xs = mesh.vert_df.loc[interior_mask, 'x'].values
    ys = mesh.vert_df.loc[interior_mask, 'y'].values
    
    def add_quiver(col_x, col_y, color, label, width=0.003, scale=25):
        if col_x in mesh.vert_df.columns:
            # Extract ONLY interior forces
            fx = mesh.vert_df.loc[interior_mask, col_x].values
            fy = mesh.vert_df.loc[interior_mask, col_y].values
            
            if np.any(np.abs(fx) > 1e-6) or np.any(np.abs(fy) > 1e-6):
                ax.quiver(xs, ys, fx, fy, color=color, scale=scale, width=width, zorder=5, label=label)

    if show_f_line:  add_quiver('f_line_x', 'f_line_y', '#22c55e', 'Line Tension')
    if show_f_area:  add_quiver('f_area_x', 'f_area_y', '#3b82f6', 'Area Pressure')
    if show_f_perim: add_quiver('f_perim_x', 'f_perim_y', '#a855f7', 'Perimeter Cont.')
    if show_f_total: add_quiver('force_x', 'force_y', '#ea580c', 'Net Force', width=0.005, scale=20)

    if show_f_line or show_f_area or show_f_perim or show_f_total:
        ax.legend(loc='upper right', fontsize=8)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax.axis('off')
    
    if standalone:
        plt.show()