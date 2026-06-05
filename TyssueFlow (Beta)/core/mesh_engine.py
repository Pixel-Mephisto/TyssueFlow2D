import numpy as np

class TissueMesh:
    def __init__(self, max_elements: int = 50000) -> None:
        self.max_elements = max_elements
        self.num_verts = 0
        self.vert_x = np.zeros(max_elements, dtype=np.float64)
        self.vert_y = np.zeros(max_elements, dtype=np.float64)
        self.force_x = np.zeros(max_elements, dtype=np.float64)
        self.force_y = np.zeros(max_elements, dtype=np.float64)
        self.is_boundary_vert = np.zeros(max_elements, dtype=np.bool_)
        
        self.num_edges = 0
        self.edge_srce = np.zeros(max_elements, dtype=np.int64)
        self.edge_trgt = np.zeros(max_elements, dtype=np.int64)
        self.edge_face = np.zeros(max_elements, dtype=np.int64)
        self.edge_next = np.zeros(max_elements, dtype=np.int64)
        self.edge_prev = np.zeros(max_elements, dtype=np.int64)
        self.edge_twin = np.zeros(max_elements, dtype=np.int64)
        
        self.num_faces = 0
        self.face_target_area = np.zeros(max_elements, dtype=np.float64)
        self.face_calc_area = np.zeros(max_elements, dtype=np.float64)
        self.face_calc_perimeter = np.zeros(max_elements, dtype=np.float64)