""" Configuration of the analysis"""
# pylint disable=invalid-name

from dataclasses import dataclass
from typing import List, Union

@dataclass
class GeneralParams:
    """General parameters of the analysis"""
    results_dir: str
    physical_length: float
    x_range: List[float]
    y_range: List[float]
    n: int

@dataclass
class FistaParams:
    """FISTA parameters"""
    num_iter: int
    num_warp: int
    alpha: float
    beta: float
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int

@dataclass
class HSParams:
    """Horn Schunck parameters"""
    num_iter: int
    num_warp: int
    alpha: float
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int
    w: float

@dataclass
class FarnebackParams:
    """Farneback parameters
    """
    winSize: int
    pyrScale: float
    numLevels: int
    fastPyramids: bool
    numIters: int
    polyN: int
    polySigma: float
    flags: int

@dataclass
class TVL1Params:
    """TV-L1 parameters
    """
    attachment: float
    tightness: float
    num_warp: int
    num_iter: int
    tol: float
    prefilter: bool

@dataclass
class ILKParams:
    """ILK parameters
    """
    radius: float
    num_warp: int
    gaussian: bool
    prefilter: bool
    
@dataclass
class OpticalFlowParams:
    """Optical Flow parameters"""
    global_flow: bool
    fista: FistaParams
    hs: HSParams
    farneback: FarnebackParams
    tvl1: TVL1Params
    ilk: ILKParams

@dataclass
class ElasticExperiment:
    """Configuration for an experiment on synthetic images of elastic cells
    """
    of_funcs: Union[List[str], str]
    vmaxstrain: float
    scale: float
    step: int
    T_for_plot: float
    E_for_plot: float
    nu_for_plot: float
    threshold_inf: float
    threshold_sup: float
    scatter_comparison: bool
    T: float | None = None
    E: float | None = None
    nu: float | None = None
    exp_ind: int | None = None
    image_id: int | None = None
    implot : int | None = None
    
@dataclass
class RegExperiment:
    """Configuration for the regularization testing experiment
    """
    of_funcs: Union[List[str], str]
    T: float
    E: float
    nu: float
    factors: List[float]
    
@dataclass
class NoiseExperiment:
    """Configuration for the noise experiment
    """
    of_funcs: Union[List[str], str]

@dataclass
class MicroExperiment:
    """Configuration for an experiment on a microscopy image
    """
    im: int
    of_funcs: Union[List[str], str]
    path: str
    active_contour: bool
    E: float
    nu: float
    vmaxstrain: float
    scale_flow: float
    step_flow: int
    scale_traction: float
    step_traction: int
    qt: bool
    vminpositions: float | None = None
    vmaxpositions: float | None = None
    alphapositions: float | None = None
    center_circle_seg: tuple[float, float] | None = None
    radius_circle_seg: float | None = None
    alpha: List[float] | None = None
    beta: List[float] | None = None
    gamma: List[float] | None = None
    
    def __post_init__(self):
        if isinstance(self.of_funcs, str):
            self.of_funcs = [self.of_funcs]