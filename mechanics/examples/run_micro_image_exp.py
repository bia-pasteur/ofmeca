"""Useful to compute mechanical quantities from a real microscopy image"""

from typing import List, Callable, Dict
from pathlib import Path
import time
import jsonargparse
import numpy as np
from mechanics.src.config import GeneralParams, OpticalFlowParams, MicroExperiment
from mechanics.src.optical_flow.algorithms import farneback, hs_of, fista_of, tv_l1, ilk
from mechanics.src.utils import compute_lame, load_order_clean, generate_mask_on_micro_image, remap
from mechanics.src.meca_of_pipeline import compute_of_strain_traction_micro_img
from mechanics.src.plot_functions import save_of_strain_traction_micro_img, plot_cell_positions

def process_image(
    image: Path, 
    maskcell: np.ndarray, 
    results_dir: Path, 
    of_for_computation: List[Callable], 
    params_for_computation: List[Dict], 
    micro_exp: MicroExperiment
) -> Dict | List[Dict]:
    """
    Process a specific simulation or experimental case by computing optical flow–based strain 
    and traction fields, optionally saving visualization plots.
    
    Args:
        results_dir (Path): Directory where results (plots and data) will be saved.
        pixel_size (float): Physical size of one pixel in the images, used for spatial scaling.
        of_for_computation (List[Callable]): List of optical flow algorithms (functions) to apply for displacement computation.
        params_for_computation (List[Dict]): List of parameter dictionaries corresponding to each optical flow method.
        micro_exp (MicroExperiment): Parameters related to the experiment, such as some plotting parameters

    Returns:
        None

    Notes:
        - The function computes the Lamé coefficients `(μ, λ)` from `(E, ν)`.  
        - Results are visualized and saved. 
        - Timing information is printed to the console for performance tracking.
    """
        
    mu, lambda_ = compute_lame(micro_exp.E, micro_exp.nu)

    start_time = time.time()
    results = compute_of_strain_traction_micro_img(
        image=image,
        mask=maskcell,
        mu=mu,
        lambda_=lambda_,
        of_functions=of_for_computation, 
        of_params=params_for_computation, 
        global_flow=False
    )
    
    save_of_strain_traction_micro_img(
        image=image,
        results=results,
        save_path=results_dir / 'plots' / f"strain_traction_plot_micro_image_{micro_exp.im}.png",
        vmaxstrain=micro_exp.vmaxstrain,
        scale_flow=micro_exp.scale_flow,
        step_flow=micro_exp.step_flow,
        scale_traction=micro_exp.scale_traction,
        step_traction=micro_exp.step_traction,
        show=False
    )
    
    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.2f} seconds")


def main(
    optical_flow: OpticalFlowParams,
    general: GeneralParams,
    micro_exp: MicroExperiment
):
    """
    Main entry point for optical flow–based strain and traction analysis on a real microscopy image

    This function orchestrates the full processing pipeline:
      1. Initializes selected optical flow methods and their parameter sets.
      2. Runs `process_case` on the chosen image

    Args:
        optical_flow (OpticalFlowParams): Configuration object containing parameter sets for each supported optical flow method
        general (GeneralParams): General configuration including paths for result storage and experiment control flags, as well as physical and spatial settings for the image domain
        micro_exp (MicroExperiment): Parameters related to the experiment, such as the path to the image file, and some plotting parameters

    Raises:
        ValueError:
            - If an unknown optical flow method name is provided in `experiment.of_funcs`.

    Returns:
        None:
            Results are saved to disk in the format of a PNG plots visualizing RMSE tables and, optionally, scatter comparisons.

    Notes:
        - The mapping of optical flow methods is defined internally:
            `"farneback"`, `"hs"`, `"tvl1"`, `"ilk"`, `"fista"`.
        - The CLI entry point is automatically created with `jsonargparse.auto_cli`,
          enabling the script to be run directly from the command line.
    """

    img, _ = load_order_clean(micro_exp.path)
    
    if micro_exp.im == 1:
        image = img[2:4, 10, 180:430, 140:390, 1]
    
    else:
        image = img[1:3, 9, 100:400, 100:400, 2]
    
    plot_cell_positions(image, vmin=micro_exp.vminpositions, vmax=micro_exp.vmaxpositions, alpha=micro_exp.alphapositions, save_path=f'{Path(general.results_dir)}/plots/cell_positions_{micro_exp.im}.png')
    
    maskcell = generate_mask_on_micro_image(image[0], active_contour=micro_exp.active_contour, center=micro_exp.center_circle_seg, radius=micro_exp.radius_circle_seg, alpha=micro_exp.alpha[0], beta=micro_exp.beta[0], gamma=micro_exp.gamma[0])

    image2 = remap(image)
    
    of_methods = {
        "farneback": (farneback, optical_flow.farneback),
        "hs":        (hs_of, optical_flow.hs),
        "tvl1":      (tv_l1, optical_flow.tvl1),
        "ilk":       (ilk, optical_flow.ilk),
        "fista":     (fista_of, optical_flow.fista),
    }
    
    of_for_computation, params_for_computation = [], []
    
    for of_func_name in micro_exp.of_funcs:
        if of_func_name not in of_methods:
            raise ValueError(f"Unknown optical flow method '{of_func_name}'")

        of_func, of_params = of_methods[of_func_name]
        of_for_computation.append(of_func)
        params_for_computation.append(of_params)
    
    process_image(image=image2, maskcell=maskcell, results_dir=Path(general.results_dir), of_for_computation=of_for_computation, params_for_computation=params_for_computation, micro_exp=micro_exp)
            
if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)

