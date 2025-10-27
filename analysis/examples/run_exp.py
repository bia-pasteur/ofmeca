"""Useful to compute mechanical quantities from a specific simulation or experimental case"""

from typing import List, Callable, Dict
from pathlib import Path
import time
import numpy as np
import pickle
import jsonargparse
from analysis.src.config import GeometryParams, GeneralParams, OpticalFlowParams, PlotParams, Experiment
from analysis.src.optical_flow.algorithms import farneback, hs_of, fista_of, tv_l1, ilk
from analysis.src.utils import compute_lame, find_experiment_folder, extract_E_from_folder, extract_T_from_folder, extract_nu_from_folder, load_images_and_displacements, results_to_df
from analysis.src.analysis_pipeline import compute_of_strain_traction
from analysis.src.plot_functions import save_of_strain_traction, save_table_rmse, save_scatter_comparison

def process_case(
    plot_parameters: PlotParams, 
    results_dir: Path, 
    pixel_size: float, 
    of_for_computation: List[Callable], 
    params_for_computation: List[Dict], 
    exp_ind=None, 
    T=None, 
    E=None, 
    nu=None, 
    image_id=None
) -> Dict | List[Dict]:
    """
    Process a specific simulation or experimental case by computing optical flow–based strain 
    and traction fields, optionally saving visualization plots.

    This function can operate in several modes:
      - **Single image mode:** if `image_id`, `T`, `E`, and `nu` are specified, it loads a specific image 
        and its ground-truth displacement field.
      - **Full parameter set mode:** if `T`, `E`, and `nu` are specified without `image_id`, it loads 
        all images corresponding to these parameters.
      - **Batch experiment mode:** if `exp_ind` is specified (1, 2, or 3), it recursively processes 
        all sub-cases of that experiment, identified by their `(T, E, nu)` folders.

    Args:
        plot_parameters (PlotParams): Plot configuration object containing visualization parameters (strain limits, quiver scale, sampling step, thresholds, etc.).
        results_dir (Path): Directory where results (plots and data) will be saved.
        pixel_size (float): Physical size of one pixel in the images, used for spatial scaling.
        of_for_computation (List[Callable]): List of optical flow algorithms (functions) to apply for displacement computation.
        params_for_computation (List[Dict]): List of parameter dictionaries corresponding to each optical flow method.
        exp_ind (int, optional): Experiment index (1, 2, or 3). If provided, the function iterates over 
            all sub-cases `(T, E, nu)` contained in the corresponding experiment folder.
        T (float, optional): Traction force magnitude used in the simulation or experiment.
        E (float, optional): Young’s modulus of the material for this case.
        nu (float, optional): Poisson’s ratio of the material for this case.
        image_id (int or str, optional): Identifier of a specific image to process within the `(T, E, nu)` case.

    Raises:
        ValueError: 
            - If only a subset of `(T, E, nu)` is provided.  
            - If `exp_ind` is not 1, 2, or 3.  
            - If required parameters are missing for a specific image.
        FileNotFoundError: 
            If the required image or displacement files do not exist in the expected folder.

    Returns:
        dict or List[dict]: 
            - If processing a single case (`image_id` or `(T, E, nu)`), returns a dictionary containing 
              the computed optical flow, strain, and traction results.  
            - If `exp_ind` is provided, returns a list of results dictionaries for all sub-cases.

    Notes:
        - The function computes the Lamé coefficients `(μ, λ)` from `(E, ν)` for each case.  
        - Results are visualized and saved only for selected cases defined by `plot_parameters`.  
        - Timing information is printed to the console for performance tracking.
    """
    base_path = Path("data")
    images = []
    displacements = []
    mu = 0
    lambda_= 0
    if image_id is not None: 
        if E is None or T is None or nu is None: 
            raise ValueError(f"Unspecified traction force T and/or Young's modulus E for simulation on image {image_id}")
        else:
            exp_folder = find_experiment_folder(base_path, T, E, nu)
            img_path = exp_folder / f"{image_id}_img.npy"
            ugt_path = exp_folder / f"{image_id}_ugt.npy"

            if not img_path.exists() or not ugt_path.exists():
                raise FileNotFoundError(f"Missing files for image_id={image_id} in {exp_folder}")

            images = [np.load(img_path)]
            displacements = [np.load(ugt_path)]
            mu, lambda_ = compute_lame(E, nu)
    
    elif (E is not None and (T is None or nu is None)) or (E is None and (T is not None or nu is not None)):
        raise ValueError("All of E, T, and nu must be specified together (either all given or all None).")
    
    elif E is not None and T is not None and nu is not None: 
        exp_folder = find_experiment_folder(base_path, T, E, nu)
        images, displacements = load_images_and_displacements(exp_folder, mode="original")
        mu, lambda_ = compute_lame(E, nu)
        
    elif exp_ind is not None: 
        if exp_ind == 1:
            exp = 'experiment_1'
        elif exp_ind == 2:
            exp = "experiment_2"
        elif exp_ind == 3:
            exp = "experiment_3"
        else : 
            raise ValueError("exp_ind can only be 1, 2 or 3")
        exp_folder = base_path / exp
        results_ = []
        for case_T_E_nu in sorted(exp_folder.iterdir()):
            E_case = extract_E_from_folder(case_T_E_nu.name)
            T_case = extract_T_from_folder(case_T_E_nu.name)
            nu_case = extract_nu_from_folder(case_T_E_nu.name)
            results_.append(process_case(plot_parameters, results_dir, pixel_size, of_for_computation, params_for_computation, T=T_case, E=E_case, nu=nu_case))
        return results_
     
    else:
        raise ValueError("Can't run with nothing specified")
    
    start_time = time.time()
    if image_id is not None: 
        print(f"\nRunning analysis on T = {T}, E = {E}, nu = {nu} for image {image_id} ...")
    elif T is not None: 
        print(f"\nRunning analysis on T = {T}, E = {E}, nu = {nu} ...")
    
    results = compute_of_strain_traction(
        images=images,
        displacements=displacements,
        pixel_size=pixel_size,
        mu=mu,
        lambda_=lambda_,
        of_functions=of_for_computation, 
        of_params=params_for_computation
    )
    
    if image_id is not None: 
        save_of_strain_traction(
            images=images,
            displacements=displacements,
            results=results,
            save_path=results_dir / 'plots' / f"strain_traction_plot_E_{E}_T_{T}_nu_{nu}_im_{image_id}.png",
            pixel_size=pixel_size,
            implot=0,
            vmaxstrain=plot_parameters.vmaxstrain,
            scale=plot_parameters.scale,
            step=plot_parameters.step,
            threshold_inf=plot_parameters.threshold_inf,
            threshold_sup=plot_parameters.threshold_sup,
            show=False
        )
    
    else: 
        if T==plot_parameters.T_for_plot and E==plot_parameters.E_for_plot and nu==plot_parameters.nu_for_plot:
            save_of_strain_traction(
                images=images,
                displacements=displacements,
                results=results,
                save_path=results_dir / 'plots' / f"strain_traction_plot_E_{E}_T_{T}_nu_{nu}_im_{plot_parameters.implot}.png",
                pixel_size=pixel_size,
                implot=plot_parameters.implot,
                vmaxstrain=plot_parameters.vmaxstrain,
                scale=plot_parameters.scale,
                step=plot_parameters.step,
                threshold_inf=plot_parameters.threshold_inf,
                threshold_sup=plot_parameters.threshold_sup,
                show=False
            )
            
    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.2f} seconds")
    
    return results

    
def main(
    plot_parameters:PlotParams,
    optical_flow:OpticalFlowParams,
    general:GeneralParams,
    geometry: GeometryParams,
    experiment: Experiment
):
    """
    Main entry point for optical flow–based strain and traction analysis.

    This function orchestrates the full processing pipeline:
      1. Initializes selected optical flow methods and their parameter sets.
      2. Computes the physical pixel size from the geometry parameters.
      3. Runs `process_case` either:
         - For all experiments (1, 2, 3) if no specific parameters are given, or
         - For a single experiment, case or image if `exp_ind` or `(T, E, ν)` and/or `image_id` are specified.
      4. Saves the computed results as serialized `.pkl` files, CSV tables, and RMSE plots.
      5. Optionally generates comparison scatter plots between experiments.

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
            - If an unknown optical flow method name is provided in `experiment.of_funcs`.
            - If experiment parameters are inconsistently defined (handled within `process_case`).
        FileNotFoundError:
            If required image or displacement files are missing for a given `(T, E, ν, image_id)`.

    Returns:
        None:
            Results are saved to disk in several formats:
            - Pickle files (`.pkl`) with raw results dictionaries.
            - CSV tables summarizing RMSE metrics per method and experiment.
            - PNG plots visualizing RMSE tables and, optionally, scatter comparisons.

    Notes:
        - The mapping of optical flow methods is defined internally:
            `"farneback"`, `"hs"`, `"tvl1"`, `"ilk"`, `"fista"`.
        - When no experiment parameters are provided, the function runs all predefined
          experiments (`exp_ind = 1, 2, 3`) sequentially.
        - The CLI entry point is automatically created with `jsonargparse.auto_cli`,
          enabling the script to be run directly from the command line.
    """

    of_methods = {
        "farneback": (farneback, optical_flow.farneback),
        "hs":        (hs_of, optical_flow.hs),
        "tvl1":      (tv_l1, optical_flow.tvl1),
        "ilk":       (ilk, optical_flow.ilk),
        "fista":     (fista_of, optical_flow.fista),
    }
    
    of_for_computation, params_for_computation = [], []
    
    for of_func_name in experiment.of_funcs:
        if of_func_name not in of_methods:
            raise ValueError(f"Unknown optical flow method '{of_func_name}'")

        of_func, of_params = of_methods[of_func_name]
        of_for_computation.append(of_func)
        params_for_computation.append(of_params)
        
    pixel_size = geometry.n / ((geometry.x_range[1]-geometry.x_range[0])*geometry.physical_length)
    
    if plot_parameters.scatter_comparison :
        dfs = []
    if all(x is None for x in (experiment.T, experiment.E, experiment.nu, experiment.exp_ind, experiment.image_id)):
        for exp_ind in [1, 2, 3]:
            results_exp = process_case(plot_parameters=plot_parameters, results_dir=Path(general.results_dir), pixel_size=pixel_size, of_for_computation=of_for_computation, params_for_computation=params_for_computation, exp_ind=exp_ind, T=experiment.T, E=experiment.E, nu=experiment.nu, image_id=experiment.image_id)
            with open(Path(general.results_dir) / 'tables_dict' / f"results_exp_{exp_ind}.pkl", "wb") as f:
                pickle.dump(results_exp, f)
            df_exp = results_to_df(results_exp)
            df_exp.to_csv(Path(general.results_dir) / 'tables_dict' / f"mean_rmse_experiment_{exp_ind}.csv", index=True)
            save_table_rmse(df_exp, Path(general.results_dir) / 'plots' / f"mean_rmse_experiment_{exp_ind}_.png")
            if plot_parameters.scatter_comparison :
                dfs.append(df_exp)
        
        if plot_parameters.scatter_comparison : 
            save_scatter_comparison(dfs, Path(general.results_dir))
            
    else: 
        results = process_case(plot_parameters=plot_parameters, results_dir=Path(general.results_dir), pixel_size=pixel_size, of_for_computation=of_for_computation, params_for_computation=params_for_computation, exp_ind=experiment.exp_ind, T=experiment.T, E=experiment.E, nu=experiment.nu, image_id=experiment.image_id)
        if experiment.image_id is not None: 
            path = f"T_{experiment.T}_E_{experiment.E}_nu_{experiment.nu}_image_{experiment.image_id}"
        else :
            path = f"T_{experiment.T}_E_{experiment.E}_nu_{experiment.nu}"
        
        with open(Path(general.results_dir) / 'tables_dict' / f'{path}.pkl', "wb") as f:
            pickle.dump(results, f)
            
        df = results_to_df(results)
        df.to_csv(Path(general.results_dir) / 'tables_dict' / f"mean_rmse_{path}.csv", index=True)
        save_table_rmse(df, Path(general.results_dir) / 'plots' / f"mean_rmse_{path}.png")
            
if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)