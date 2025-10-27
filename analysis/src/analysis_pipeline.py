"""Useful to get the results of the mechanical computations on images stored in a dictionary"""
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
from typing import List, Callable, Dict
import numpy as np
import scipy.ndimage as ndi
from analysis.src.utils import rmse
from analysis.src.mechanics.quantities_computation import (
    strain_mask, deformation, stress_mask,
    compute_normals_from_mask_2d, compute_traction_2d
)

def compute_of_strain_traction(
    images: List[np.ndarray], 
    displacements: List[np.ndarray], 
    pixel_size: float, 
    mu: float,
    lambda_: float, 
    of_functions: List[Callable],
    of_params: List[Dict]
) -> Dict: 
    """
    Compute optical-flow-based displacement, strain, deformation, stress, and traction fields,
    and compare them to ground-truth quantities for one or several images.

    This function evaluates several optical flow (OF) methods on a sequence of images with 
    known ground-truth displacements. For each image, it computes the corresponding
    strain, deformation gradient, stress, and traction fields, and calculates the 
    root mean square error (RMSE) between the OF-based estimates and the ground truth.
    Mean RMSE values across all images are also provided.

    Args:
        images (List[np.ndarray]): List of 2D grayscale images (float or uint) used as inputs to optical flow methods.
        displacements (List[np.ndarray]): List of ground-truth displacement fields for each image.
        pixel_size (float):Conversion factor from pixel to physical units (e.g., µm/pixel).
        mu (float): Lamé parameter
        lambda_ (float): Lamé parameter
        of_functions (List[Callable]): List of optical flow functions to evaluate. 
        of_params (List[Dict]): List of parameter dictionaries corresponding to each function in `of_functions`.

    Returns
        dict:
            Dictionary containing, for each image index:
            
            - `"flows"`, `"strain"`, `"deformation"`, `"stress"`, `"traction"`:
            dictionaries with ground-truth (`"gt"`) and OF-based results per method.
            - `"rmse_flows"`, `"rmse_strain"`, `"rmse_def"`, `"rmse_stress"`, `"rmse_traction"`:
            RMSE values comparing OF estimates to ground truth.
            - `"mask"` : binary mask used for valid regions.
            
            If multiple images are provided, the following mean metrics are also computed:
            - `"mean_rmse_disp"`, `"mean_rmse_strain"`, `"mean_rmse_def"`, 
            `"mean_rmse_stress"`, `"mean_rmse_traction"`.
    
    """
    
    results = {}
    
    for nb, image in enumerate(images):
        results[nb] = {}
        displacement = displacements[nb]
        disp_gt = displacement * pixel_size
        mask = (disp_gt[0,0] != 0)
        strain_gt = strain_mask(displacement, [1/pixel_size, 1/pixel_size], mask)
        def_gt = deformation(strain_gt)
        stress_gt = stress_mask(strain_gt, mu, lambda_)
        mask_eroded_gt = ndi.binary_erosion(mask)
        inner_boundary_gt = mask_eroded_gt & (~ndi.binary_erosion(mask_eroded_gt))
        normals_gt = compute_normals_from_mask_2d(mask_eroded_gt)
        normals_gt[:, ~inner_boundary_gt] = 0
        traction_gt = compute_traction_2d(stress_gt[:,:,0], -normals_gt)
        
        results[nb]["flows"] = {"gt": disp_gt}
        results[nb]["strain"] = {"gt": strain_gt}
        results[nb]["deformation"] = {"gt": def_gt}
        results[nb]["stress"] = {"gt": stress_gt}
        results[nb]["traction"] = {"gt": traction_gt}
        results[nb]["mask"] = mask
        results[nb]["rmse_flows"] = {}
        results[nb]["rmse_strain"] = {}
        results[nb]["rmse_def"] = {}
        results[nb]["rmse_stress"] = {}
        results[nb]["rmse_traction"] = {}
        
        for i, method in enumerate(of_functions): 
            method_name = method.__name__.replace("_of", "")
            h = method(image, of_params[i])
            h_mask = h * mask
            rmse_flow = rmse(h_mask, disp_gt)
            
            strain_of = strain_mask(h_mask, [1, 1], mask)
            rmse_strain = rmse(strain_of, strain_gt)
            
            def_of = deformation(strain_of)
            rmse_def = rmse(def_of, def_gt)
            
            stress_of = stress_mask(strain_of, mu, lambda_)
            rmse_stress = rmse(stress_of, stress_gt)

            traction_of = compute_traction_2d(stress_of[:,:,0], -normals_gt)
            rmse_traction = rmse(traction_of, traction_gt)

            results[nb]["flows"][method_name] = h_mask
            results[nb]["strain"][method_name] = strain_of
            results[nb]["deformation"][method_name] = def_of
            results[nb]["stress"][method_name] = stress_of
            results[nb]["traction"][method_name] = traction_of
            
            results[nb]["rmse_flows"][method_name] = rmse_flow
            results[nb]["rmse_strain"][method_name] = rmse_strain
            results[nb]["rmse_def"][method_name] = rmse_def
            results[nb]["rmse_stress"][method_name] = rmse_stress
            results[nb]["rmse_traction"][method_name] = rmse_traction

    if len(images)>1:
        results["mean_rmse_disp"] = {}
        results["mean_rmse_strain"] = {}
        results["mean_rmse_def"] = {}
        results["mean_rmse_stress"] = {}
        results["mean_rmse_traction"] = {}
        
        for method in of_functions:
            m = method.__name__.replace("_of", "")
            disp_vals, strain_vals, def_vals, stress_vals, trac_vals = [], [], [], [], []
            for nb, res in results.items():
                if not isinstance(nb, int): 
                    continue
                if res:
                    disp_vals.append(res["rmse_flows"][m])
                    strain_vals.append(res["rmse_strain"][m])
                    def_vals.append(res["rmse_def"][m])
                    stress_vals.append(res["rmse_stress"][m])
                    trac_vals.append(res["rmse_traction"][m])
                    
            results["mean_rmse_disp"][m] = np.mean(disp_vals)
            results["mean_rmse_strain"][m] = np.mean(strain_vals)
            results["mean_rmse_def"][m] = np.mean(def_vals)
            results["mean_rmse_stress"][m] = np.mean(stress_vals)
            results["mean_rmse_traction"][m] = np.mean(trac_vals)

    return results