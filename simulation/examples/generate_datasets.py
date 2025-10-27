"""Useful to generate synthetic images of deforming cells"""
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
import os
import numpy as np
import jsonargparse
from simulation.src.imaging.generator import dirichlet
from simulation.src.config import GeneralParams, GeometryParams, MaterialParams, ExperimentsParams, ImagingParams
from simulation.src.mesh.creation import gmsh_cell_shape
from simulation.src.simulation_pipeline import create_displacement_image

def main(
    general:GeneralParams, 
    geom: GeometryParams, 
    mat: MaterialParams,
    exp:ExperimentsParams, 
    img: ImagingParams
):
    """
    Generates simulated displacement fields and corresponding images for a set of experiments.

    This function runs simulations across different parameter combinations for traction 
    force (T), Young’s modulus (E), and Poisson’s ratio (ν), generating synthetic 
    images and displacement fields. Results are saved as `.npy` files in structured 
    experiment directories.

    Args:
        general (GeneralParams): General simulation parameters including:
              - `dataset_dir`: Base directory for saving generated data.
              - `t_end`: Total simulation time.
              - `num_time_steps`: Number of time steps.
        geom (GeometryParams): Geometrical parameters of the simulation, including:
              - `physical_length`: Physical size scaling factor.
              - `pipette_radius_factor`: Radius factor for pipette.
              - `pipette_center`: Coordinates of pipette center.
              - `x_range`, `y_range`: Spatial ranges for the simulation domain.
              - `n`: Spatial discretization parameter.
        mat (MaterialParams): Material properties, e.g., viscosity `eta`.
        exp (ExperimentsParams): Experiment configuration parameters, including:
              - `traction_zone`: List of traction force values for experiment 1.
              - `ym_for_t_nu`: Young’s modulus for experiment 1.
              - `nu_for_ym_t`: Poisson’s ratio for experiment 1 and 2.
              - `t_for_ym_nu`: Traction force for experiments 2 and 3.
              - `youngs_modulus`: List of Young’s modulus values for experiment 2.
              - `nu`: List of Poisson’s ratio values for experiment 3.
              - `seeds`: Random seeds for reproducibility.
        img (ImagingParams): Imaging parameters, including:
              - `num_points`: Number of image points.
              - `noise_amplitude`: Amplitude of noise to add to images.
              - `num_fourier_modes`: Number of Fourier modes for image generation.
              - `lc`: Correlation length for random fields.
              - `grain`: Grain size for image generation.

    Raises:
        FileNotFoundError:
            If any of the required output directories cannot be created (e.g., insufficient permissions).

    Notes:
        Output files are saved in directories following the convention `experiment_{n}/T_{T}_E_{E}_nu_{nu}/`
        in files and `<seed>_img.npy` for images and `<seed>_ugt.npy` for displacement fields
    """
    pipette_radius = geom.physical_length * geom.pipette_radius_factor
            
    ym = float(exp.ym_for_t_nu)
    nu = exp.nu_for_ym_t
    for tzone in exp.traction_zone:
        tzone = float(tzone)
        for seed in exp.seeds:
            name = f"{seed}"
            print(f"\n Running simulation {name} for T={tzone}, E={ym} and nu={nu}")

            u_gt, img_final = create_displacement_image(
                gmsh_mesh=gmsh_cell_shape,
                physical_length=geom.physical_length,
                dirichlet=dirichlet,
                t_end=general.t_end,
                num_time_steps=general.num_time_steps,
                zone_radius=pipette_radius,
                zone_center=tuple(geom.pipette_center[:2]),
                traction_zone=tzone,
                youngs_modulus=ym,
                nu=nu,
                eta=mat.eta,
                n=geom.n,
                name=name,
                x_range=tuple(geom.x_range),
                y_range=tuple(geom.y_range),
                num_points=img.num_points,
                noise_amplitude=img.noise_amplitude,
                num_fourier_modes=img.num_fourier_modes,
                lc=img.lc,
                grain=img.grain,
                seed=seed
            )
            
            output_dir = os.path.join(general.dataset_dir, f"experiment_1/T_{tzone}_E_{ym}_nu_{nu}")
            os.makedirs(output_dir, exist_ok=True)

            np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
            np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)

    tzone = float(exp.t_for_ym_nu)
    nu = exp.nu_for_ym_t
    for ym in exp.youngs_modulus:
        ym = float(ym)
        for seed in exp.seeds:
            name = f"{seed}"
            print(f"\n Running simulation {name} for T={tzone}, E={ym} and nu={nu}")

            u_gt, img_final = create_displacement_image(
                gmsh_mesh=gmsh_cell_shape,
                physical_length=geom.physical_length,
                dirichlet=dirichlet,
                t_end=general.t_end,
                num_time_steps=general.num_time_steps,
                zone_radius=pipette_radius,
                zone_center=tuple(geom.pipette_center[:2]),
                traction_zone=tzone,
                youngs_modulus=ym,
                nu=nu,
                eta=mat.eta,
                n=geom.n,
                name=name,
                x_range=tuple(geom.x_range),
                y_range=tuple(geom.y_range),
                num_points=img.num_points,
                noise_amplitude=img.noise_amplitude,
                num_fourier_modes=img.num_fourier_modes,
                lc=img.lc,
                grain=img.grain,
                seed=seed
            )
            output_dir = os.path.join(general.dataset_dir, f"experiment_2/T_{tzone}_E_{ym}_nu_{nu}")
            os.makedirs(output_dir, exist_ok=True)  

            np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
            np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)
    
    tzone = float(exp.t_for_ym_nu)
    ym = float(exp.ym_for_t_nu)
    for nu in exp.nu:
        for seed in exp.seeds:
            name = f"{seed}"
            print(f"\n Running simulation {name} for T={tzone}, E={ym} and nu={nu}")

            u_gt, img_final = create_displacement_image(
                gmsh_mesh=gmsh_cell_shape,
                physical_length=geom.physical_length,
                dirichlet=dirichlet,
                t_end=general.t_end,
                num_time_steps=general.num_time_steps,
                zone_radius=pipette_radius,
                zone_center=tuple(geom.pipette_center[:2]),
                traction_zone=tzone,
                youngs_modulus=ym,
                nu=nu,
                eta=mat.eta,
                n=geom.n,
                name=name,
                x_range=tuple(geom.x_range),
                y_range=tuple(geom.y_range),
                num_points=img.num_points,
                noise_amplitude=img.noise_amplitude,
                num_fourier_modes=img.num_fourier_modes,
                lc=img.lc,
                grain=img.grain,
                seed=seed
            )
            output_dir = os.path.join(general.dataset_dir, f"experiment_3/T_{tzone}_E_{ym}_nu_{nu}")
            os.makedirs(output_dir, exist_ok=True)  

            np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
            np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)

if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)