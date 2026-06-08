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
    p_0: float = 0
    Lambda: float = 1.0
    dt: float = 0.05
    brownian_mag: float = 0.1
    use_brownian: bool = True

@dataclass
class T1Config:
    threshold: float = 0.05
    rest_length: float = 0.08

@dataclass
class BiochemConfig:
    gN: float = 1400.0     # Notch production rate
    gD: float = 1600.0     # Delta production rate
    kC: float = 5e-4       # Cis-inhibition rate
    gamma: float = 0.1     # N/D degradation rate
    gammaI: float = 0.5    # NICD degradation rate
    I0: float = 200.0      # Hill function reference threshold
    kT0: float = 5e-5      # Baseline Trans-activation rate
    
    dt_chem: float = 0.1   # Biochemical time step
    steps_per_mech: int = 5 # Run 5 chem steps per 1 physics step