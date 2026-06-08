from dataclasses import dataclass

@dataclass
class MeshConfig:
    grid_type: str = "hex"
    nx: int = 8
    ny: int = 8
    noise_percent: float = 25.0

@dataclass
class PhysicsConfig:
    K_A: float = 1.0
    Gamma_P: float = 0.1
    p_0: float = 3.81
    Lambda: float = 1.0
    dt: float = 0.05
    brownian_mag: float = 0.1
    use_brownian: bool = False

@dataclass
class T1Config:
    threshold: float = 0.05
    rest_length: float = 0.08