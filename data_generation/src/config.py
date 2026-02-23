"""Config"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class GeneralParams:
    """General parameters of the simulations
    """
    dataset_dir: str
    physical_length: float
    pipette_radius_factor: float
    pipette_center: Tuple[float, float, float]
    x_range: List[float]
    y_range: List[float]
    n: int
    num_points: int
    noise_amplitude: float
    num_fourier_modes: int
    lc: float
    grain: int

@dataclass
class ElasticSimuParams:
    """Parameters for the elastic cell simulation
    """
    t_end: float 
    num_time_steps: int 
    eta: float
    ym_for_t_nu: float
    t_for_ym_nu: float
    nu_for_ym_t: float
    traction_zone: List[float]
    youngs_modulus: List[float]
    nu: List[float]
    seeds: List[int]
    
@dataclass
class NoiseSimuParams:
    """Parameters for the noisy images
    """
    traction_zone: float
    youngs_modulus: float
    nu: float
    seed: int
    noise_stds: List[float]