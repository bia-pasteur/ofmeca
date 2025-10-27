""" Configuration of the analysis"""
# pylint disable=invalid-name

from dataclasses import dataclass
from typing import List, Union

@dataclass
class GeneralParams:
    """General parameters of the analysis"""
    results_dir: str

@dataclass
class GeometryParams:
    """Geometry parameters of the cell"""
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
class PlotParams:
    """Plotting parameters"""
    implot: int
    vmaxstrain: float
    scale: float
    step: int
    T_for_plot: float
    E_for_plot: float
    nu_for_plot: float
    threshold_inf: float
    threshold_sup: float
    scatter_comparison: bool
    plot_noise: bool
    plot_reg: bool

@dataclass
class OpticalFlowParams:
    """Optical Flow parameters"""
    fista: FistaParams
    hs: HSParams
    farneback: FarnebackParams
    tvl1: TVL1Params
    ilk: ILKParams

@dataclass
class Experiment:
    """Configuration for single image and optical flow function analysis
    """
    of_funcs: Union[List[str], str]
    T: float | None = None
    E: float | None = None
    nu: float | None = None
    exp_ind: int | None = None
    image_id: int | None = None
    factors_for_reg: List[float] | None = None
    T_for_reg: float | None = None
    E_for_reg: float | None = None
    nu_for_reg: float | None = None
    
    def __post_init__(self):
        if isinstance(self.of_funcs, str):
            self.of_funcs = [self.of_funcs]