"""Useful to run the noise sensitivity analysis"""

from typing import List, Callable, Dict
from pathlib import Path
import time
import pickle
from collections import defaultdict
import jsonargparse
import numpy as np 
from mechanics.src.config import GeneralParams, OpticalFlowParams, NoiseExperiment
from mechanics.src.optical_flow.algorithms import farneback, hs_of, fista_of, tv_l1, ilk
from mechanics.src.utils import compute_lame, extract_E_from_folder, extract_nu_from_folder, load_images_and_displacements, get_all_stds_from_folder
from mechanics.src.meca_of_pipeline import compute_of_strain_traction
from mechanics.src.plot_functions import plot_mean_error_noise


def process_noise(
    noise_path:Path,
    mu:float,
    lambda_:float,
    pixel_size: float,
    of_for_computation: List[Callable],
    params_for_computation: List[Dict], 
    global_flow: bool,
) -> Dict:
    """
    Runs a noise sensitivity analysis for optical flow-based strain and traction estimation.

    This function tests how mechanical computations are affected by noise.It loads a reference noisy 
    image and corresponding ground truth displacement field, and computes RMSE metrics for each method 
    and scaling factor.

    Args:
        noise_path (Path): Path to the experiment folder containing the increasingly versions of the same image.
        mu (float): Lamé parameter
        lambda_ (float): Lamé parameter 
        pixel_size (float): Size of one pixel in physical units, used for spatial scaling.
        of_for_computation (List[Callable]): List of optical flow functions to evaluate. 
        params_of (List[Dict]): List of parameter objects or dictionaries corresponding to each optical flow method in `of_for_computation`.

    Raises:
        FileNotFoundError: 
            If `noise_path` doesn't exist.

    Returns:
        Dict: 
            Nested dictionary containing RMSE values for each optical flow method and noise level.
            Structure:
            ```
            {
                'flow': {method_name: [rmse_values_per_increasing_noise_level]},
                'deformation': {method_name: [rmse_values_increasing_noise_level]},
                'traction': {method_name: [rmse_values_increasing_noise_level]}
            }
            ```
            - `flow`: RMSE of the optical flow fields.
            - `deformation`: RMSE of the reconstructed deformation fields.
            - `traction`: RMSE of the estimated traction fields.
    """
    print("\nRunning noise analysis on a new image...")
    start_time = time.time()
    images, displacements = load_images_and_displacements(noise_path, mode="noisy")
    
    results = compute_of_strain_traction(
        images=images,
        displacements=displacements,
        pixel_size=pixel_size,
        mu=mu,
        lambda_=lambda_,
        of_functions=of_for_computation, 
        of_params=params_for_computation,
        global_flow=global_flow
    )
    elapsed = time.time() - start_time
    print(f"Noise analysis completed in {elapsed:.2f} seconds")
    return results


def main(
    optical_flow: OpticalFlowParams,
    general: GeneralParams,
    noise_exp: NoiseExperiment
):
    """
    Runs a noise sensitivity analysis for optical flow-based strain and traction estimation.

    Args:
        plot_parameters (PlotParams): Configuration for visualization and result plotting 
        optical_flow (OpticalFlowParams): Configuration object containing parameter sets for each supported optical flow method
        general (GeneralParams): General configuration including paths for result storage and experiment control flags.
        geometry (GeometryParams): Physical and spatial settings of the image domain, including the number of pixels (`n`), 
            the physical domain range (`x_range`), and total physical length (`physical_length`),
            used to compute `pixel_size`.
        experiment (Experiment): Definition of which case(s) to process, including traction force `T`,
            Young’s modulus `E`, Poisson’s ratio `ν`, experiment index `exp_ind`,
            and optionally a specific `image_id`.
            Also includes the list `of_funcs` specifying which optical flow methods to run.
    Raises:
        ValueError: 
            If the optical flow functions provided in the experiment class are not in the available optical flow functions.
    """

    of_methods = {
        "farneback": (farneback, optical_flow.farneback),
        "hs":        (hs_of, optical_flow.hs),
        "tvl1":      (tv_l1, optical_flow.tvl1),
        "ilk":       (ilk, optical_flow.ilk),
        "fista":     (fista_of, optical_flow.fista),
    }
    
    of_for_computation, params_for_computation = [], []
    
    for of_func_name in noise_exp.of_funcs:
        if of_func_name not in of_methods:
            raise ValueError(f"Unknown optical flow method '{of_func_name}'")

        of_func, of_params = of_methods[of_func_name]
        of_for_computation.append(of_func)
        params_for_computation.append(of_params)
        
    pixel_size = general.n / ((general.x_range[1]-general.x_range[0])*general.physical_length)
    
    base_path = Path("data")
    noise_folder = next(base_path.glob("noise_experiment*"))
    case_name = noise_folder.name
    E = extract_E_from_folder(case_name)
    nu = extract_nu_from_folder(case_name)
    mu, lambda_ = compute_lame(E, nu)
    
    rmse_acc_noise = defaultdict(lambda: defaultdict(list))

    ind = 0
    for noise_path in sorted(noise_folder.iterdir()):
        if not noise_path.is_dir():
            continue
        stds = get_all_stds_from_folder(noise_path)
        results = process_noise(
            noise_path=noise_path,
            mu=mu, 
            lambda_=lambda_,
            pixel_size=pixel_size,
            of_for_computation=of_for_computation,
            params_for_computation=params_for_computation, 
            global_flow=optical_flow.global_flow
        )
        
        with open(Path(general.results_dir) / 'tables_dict' / f"results_noise_{ind}.pkl", "wb") as f:
            pickle.dump(results, f)
        ind += 1
        
        method_names = []
        for _, method in enumerate(of_for_computation): 
            method_names.append(method.__name__.replace("_of", ""))
            
        rmse_dict = {}
        rmse_dict['flow'] = {}
        rmse_dict['deformation'] = {}
        rmse_dict['traction'] = {}
        
        for m in method_names:
            rmse_dict['flow'][m] = []
            rmse_dict['deformation'][m] = []
            rmse_dict['traction'][m] = []
        
        for nb, res in results.items():
            if not isinstance(nb, int):
                continue
            else:
                for m in method_names:
                    rmse_dict['flow'][m].append(res['rmse_flows'][m])
                    rmse_dict['deformation'][m].append(res['rmse_def'][m])
                    rmse_dict['traction'][m].append(res['rmse_traction'][m])
        
        for key in ['deformation', 'traction']:
            for method in rmse_dict[key]:
                rmse_acc_noise[key][method].append(rmse_dict[key][method])
                
        rmse_mean_noise = {key: {method: np.mean(rmse_acc_noise[key][method], axis=0)
                                for method in rmse_acc_noise[key]}
                        for key in rmse_acc_noise}
        
        plot_mean_error_noise(rmse_mean_noise, stds, Path(general.results_dir))

if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)