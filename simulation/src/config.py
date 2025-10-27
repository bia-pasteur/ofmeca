"""Config"""

from dataclasses import dataclass
from typing import List, Tuple
import yaml

@dataclass
class GeneralParams:
    """General parameters of the simulations
    """
    dataset_dir: str
    t_end: int
    num_time_steps: int

@dataclass
class GeometryParams:
    """Geometry parameters of the cell
    """
    physical_length: float
    pipette_radius_factor: float
    pipette_center: Tuple[float, float, float]
    x_range: List[float]
    y_range: List[float]
    n: int

@dataclass
class MaterialParams:
    """Material parameters
    """
    nu: float
    eta: float

@dataclass
class ExperimentsParams:
    """Experiment parameters
    """
    ym_for_t_nu: float
    t_for_ym_nu: float
    nu_for_ym_t: float
    traction_zone: List[float]
    youngs_modulus: List[float]
    nu: List[float]
    youngs_modulus: List[float]
    seeds: List[int]
    T_for_noisy_exp: float
    E_for_noisy_exp: float
    nu_for_noisy_exp: float
    seed_for_noisy_exp: int
    noise_stds: List[float]

@dataclass
class ImagingParams:
    """_summary_
    """
    num_points: int
    noise_amplitude: float
    num_fourier_modes: int
    lc: float
    grain: int

@dataclass
class SimulationConfig:
    """_summary_
    """
    general: GeneralParams
    geometry: GeometryParams
    material: MaterialParams
    experiments: ExperimentsParams
    imaging: ImagingParams