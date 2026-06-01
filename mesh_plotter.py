import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mesh_engine import TissueMesh

def plot_mesh(mesh: TissueMesh, 
              show_faces: bool = True, 
              show_edges: bool = True, 
              show_verts: bool = True, 
              show_labels: bool = True, 
              title: str = "Tissue Topology") -> None:
    """
    Renders the half-edge dataframe architecture cleanly.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f8fafc')
    
    num_edges = len(mesh.edge_df)
    render_text = show_labels and num_edges < 200 
    face_centroids = {}
    
    # --- 1. RENDER FACES ---
    for face_id in mesh.face_df.index:
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
                
        face_centroids[face_id] = np.mean(poly_coords, axis=0)
        
        if show_faces:
            polygon = plt.Polygon(poly_coords, facecolor='#bae6fd', alpha=0.5, edgecolor='none', zorder=1)
            ax.add_patch(polygon)
        
        if render_text:
            cx, cy = face_centroids[face_id]
            ax.text(cx, cy, f"F{face_id}", ha='center', va='center', weight='bold', color='#0284c7', fontsize=12)

    # --- 2. RENDER HALF-EDGES ---
    shift_dist = 0.05
    shrink = 0.10

    if show_edges:
        for edge_id, edge_row in mesh.edge_df.iterrows():
            v1, v2 = mesh.vert_df.loc[edge_row['srce']], mesh.vert_df.loc[edge_row['trgt']]
            x1, y1, x2, y2 = v1['x'], v1['y'], v2['x'], v2['y']
            
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length < 1e-5: continue  # Safe division check
            
            fc_x, fc_y = face_centroids[edge_row['face']]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            to_center_x, to_center_y = fc_x - mx, fc_y - my
            
            nx, ny = -dy / length, dx / length
            if (nx * to_center_x + ny * to_center_y) < 0:
                nx, ny = -nx, -ny

            x1_s = x1 + nx * shift_dist + (dx / length) * shrink
            y1_s = y1 + ny * shift_dist + (dy / length) * shrink
            x2_s = x2 + nx * shift_dist - (dx / length) * shrink
            y2_s = y2 + ny * shift_dist - (dy / length) * shrink
            
            arrow = patches.FancyArrowPatch(
                (x1_s, y1_s), (x2_s, y2_s),
                arrowstyle='-|>', mutation_scale=12,
                color='#ef4444', lw=1.2, zorder=2, alpha=0.85
            )
            ax.add_patch(arrow)
            
            if render_text:
                label_shift = 0.18 
                lx, ly = mx + nx * label_shift, my + ny * label_shift
                ax.text(lx, ly, f"e{edge_id}", color='#991b1b', fontsize=8, 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))

    # --- 3. RENDER VERTICES & BOUNDARIES ---
    for edge_id, edge_row in mesh.edge_df.iterrows():
        v1, v2 = mesh.vert_df.loc[edge_row['srce']], mesh.vert_df.loc[edge_row['trgt']]
        ax.plot([v1['x'], v2['x']], [v1['y'], v2['y']], color='#94a3b8', lw=1.0, zorder=1)

    if show_verts:
        ax.scatter(mesh.vert_df['x'], mesh.vert_df['y'], color='#0f172a', s=45, zorder=3)
        if render_text:
            for v_id, v_row in mesh.vert_df.iterrows():
                ax.text(v_row['x'] - 0.05, v_row['y'] + 0.05, f"v{v_id}", color='black', weight='bold', fontsize=10)

    ax.set_aspect('equal')
    ax.set_title(title, pad=15, weight='bold', fontsize=14, color='#0f172a')
    ax.axis('off')
    plt.tight_layout()
    plt.show()