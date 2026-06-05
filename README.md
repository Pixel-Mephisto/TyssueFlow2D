# TyssueFlow2D

TyssueFlow2D is a research-grade simulation framework designed for modeling the biophysics of epithelial tissue. By leveraging Numba-accelerated C-level physics and a decoupled Model-View-Controller architecture, TyssueFlow2D provides a high-fidelity environment for simulating vertex-based tissue mechanics, cell intercalation (T1 transitions), and biochemical signaling.

⚙️ Architecture Design
TyssueFlow2D is architected for maximum separation of concerns, allowing researchers to modify biological logic or physics without needing to touch the rendering or UI code.

core/: Manages the "Dumb" data containers. It uses a Struct-of-Arrays (SoA) approach instead of object-heavy structures, ensuring minimal memory overhead and maximum cache performance for large tissues.

model/: The "Academic Sandbox." This layer holds pure mathematical physics (physics.py) and parameter configurations (config.py). It is completely environment-agnostic—you can run these simulations in a GUI, a Jupyter notebook, or a headless HPC cluster.

view/: The visual presenter. Implements "Presenter Mode" UI, with optimized Matplotlib rendering and high-contrast, scalable widgets for screen sharing and live presentation.

controller/: The logic bridge. Orchestrates the flow between UI commands, the mechanical solver, and data export pipelines.

🚀 Key Functionalities
1. High-Performance Mechanics Engine
Numba Parallelization: Physics loops are pre-compiled into machine code, bypassing Python's performance bottlenecks.

Langevin Dynamics: Supports stochastic Brownian motion for realistic membrane fluctuations.

Vertex-Model Hamiltonian: Computes energy gradients for Area Elasticity, Perimeter Contractility, and Line Tension simultaneously.

2. Topological Surgery
Automated T1 Transitions: Real-time identification and resolution of shrinking edges using a sophisticated Half-Edge data structure. The solver dynamically performs "pointer surgery" on the mesh to maintain tissue integrity during active relaxation.

Dynamic Resolution: The engine allows for customizable topological thresholds and resolved edge lengths to prevent mesh singularities and numerical explosions.

3. Presenter-Ready UI
Interactive Controls: Precision-stepped sliders for all mechanical parameters (K 
A
​
 , Γ 
P
​
 , p 
0
​
 , Adhesion Ratio).

Live Telemetry: Real-time energy invariant plotting with LaTeX-rendered Hamiltonian display.

Video Pipeline: Seamless in-application video export (MP4) to capture complex tissue dynamics for publications and presentations.

📐 Numerical Methodology
The framework employs an Overdamped Viscous Solver where node displacement is defined as dx= 
η
F⋅dt
​
 .

Normalization
The framework supports both Square and Hexagonal lattices. To ensure biological consistency, the factory automatically scales the initial lattice geometry so that a perfect cell possesses a target area (A 
0
​
 ) of exactly 1.0, regardless of the tiling geometry selected.

🛠 Usage
To launch the simulation, navigate to the project root and execute:

Bash
python main.py
Extending the Model
To add new physics: Modify model/physics.py to add new Hamiltonian energy terms.

To add chemical signaling: Implement ODE systems in a new file within the model/ folder and bridge them to the TissueMesh property arrays.

To expand topology: The core/mesh_engine.py is pre-allocated with pre-sized memory blocks, making the implementation of Cell Division (Mitosis) or Apoptosis (Extrusion) a straightforward task of array manipulation.

Quick Tips for the PI/Collaborators
Adhesion Ratio (p): We define the effective shape index ratio p= 
2Γ 
P
​
 
Λ
​
 . Increasing this value will fluidize the tissue, while decreasing it will promote a solid, elastic phenotype.

Noise Scaling: Structural noise is automatically scaled relative to the lattice side length (s≈0.62 for hex), ensuring that the same noise percentage yields consistent results across different mesh geometries.
