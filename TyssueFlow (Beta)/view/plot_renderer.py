import matplotlib.patches as patches

def render_mesh_to_axis(mesh, ax, show_forces=False, title="Tissue Workspace"):
    bg_color = '#1e1e1e'; face_color = '#2d2d30'; edge_color = '#4fc1ff'
    vert_color = '#ff6b6b'; text_color = '#d4d4d4'; arrow_color = '#facc15'
    
    ax.clear()
    ax.set_facecolor(bg_color)
    
    # 1. RENDER BOLD FACES
    for f in range(mesh.num_faces):
        start_edge = -1
        for e in range(mesh.num_edges):
            if mesh.edge_face[e] == f:
                start_edge = e; break
        if start_edge == -1: continue
        
        curr_edge = start_edge; poly_coords = []; visited = set()
        while curr_edge not in visited:
            visited.add(curr_edge)
            v = mesh.edge_srce[curr_edge]
            poly_coords.append([mesh.vert_x[v], mesh.vert_y[v]])
            curr_edge = mesh.edge_next[curr_edge]
            if curr_edge == start_edge: break
            
        if len(poly_coords) > 2:
            poly = patches.Polygon(poly_coords, closed=True, facecolor=face_color, edgecolor=edge_color, lw=2.0, alpha=0.85)
            ax.add_patch(poly)

    # 2. RENDER THICK EDGES
    for e in range(mesh.num_edges):
        v1, v2 = mesh.edge_srce[e], mesh.edge_trgt[e]
        ax.plot([mesh.vert_x[v1], mesh.vert_x[v2]], [mesh.vert_y[v1], mesh.vert_y[v2]], color=edge_color, lw=3.5, zorder=1, alpha=0.8)

    # 3. RENDER LARGE VERTICES
    if mesh.num_verts > 0:
        ax.scatter(mesh.vert_x[:mesh.num_verts], mesh.vert_y[:mesh.num_verts], color=vert_color, s=90, zorder=3, edgecolors='white', linewidths=1.5)

    # 4. RENDER LIVE FORCES
    if show_forces:
        for v in range(mesh.num_verts):
            if not mesh.is_boundary_vert[v]:
                fx, fy = mesh.force_x[v], mesh.force_y[v]
                if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                    ax.quiver(mesh.vert_x[v], mesh.vert_y[v], fx, fy, color=arrow_color, scale=15, width=0.006, zorder=5)

    ax.relim(); ax.autoscale_view(); ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=16, fontweight='bold', color=text_color, pad=15)
    ax.axis('off')