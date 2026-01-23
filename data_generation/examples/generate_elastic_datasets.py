"""Useful to generate synthetic images of deforming cells"""
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
import os
import numpy as np
import jsonargparse
from data_generation.src.imaging.generator import dirichlet
from data_generation.src.config import GeneralParams, ElasticSimuParams 
from data_generation.src.mesh.creation import gmsh_cell_shape
from data_generation.src.simulation_pipeline import create_displacement_image

def main(
    general:GeneralParams, 
    elastic_simu:ElasticSimuParams
):
    """
    Generates simulated displacement fields and corresponding images for a set of experiments.

    This function runs simulations across different parameter combinations for traction 
    force (T), Young’s modulus (E), and Poisson’s ratio (ν), generating synthetic 
    images and displacement fields. Results are saved as `.npy` files in structured 
    experiment directories.

    Args:
        general (GeneralParams): General simulation parameters
        elastic_simu (ExperimentsParams): Experiment configuration parameters

    Raises:
        FileNotFoundError:
            If any of the required output directories cannot be created (e.g., insufficient permissions).

    Notes:
        Output files are saved in directories following the convention `experiment_{n}/T_{T}_E_{E}_nu_{nu}/`
        in files and `<seed>_img.npy` for images and `<seed>_ugt.npy` for displacement fields
    """
    pipette_radius = general.physical_length * general.pipette_radius_factor
            
    ym = float(elastic_simu.ym_for_t_nu)
    nu = elastic_simu.nu_for_ym_t
    for tzone in elastic_simu.traction_zone:
        tzone = float(tzone)
        for seed in elastic_simu.seeds:
            name = f"{seed}"
            print(f"\n Running simulation {name} for T={tzone}, E={ym} and nu={nu}")

            u_gt, img_final = create_displacement_image(
                gmsh_mesh=gmsh_cell_shape,
                physical_length=general.physical_length,
                dirichlet=dirichlet,
                t_end=elastic_simu.t_end,
                num_time_steps=elastic_simu.num_time_steps,
                zone_radius=pipette_radius,
                zone_center=tuple(general.pipette_center[:2]),
                traction_zone=tzone,
                youngs_modulus=ym,
                nu=nu,
                eta=elastic_simu.eta,
                n=general.n,
                name=name,
                x_range=tuple(general.x_range),
                y_range=tuple(general.y_range),
                num_points=general.num_points,
                noise_amplitude=general.noise_amplitude,
                num_fourier_modes=general.num_fourier_modes,
                lc=general.lc,
                grain=general.grain,
                seed=seed
            )
            
            output_dir = os.path.join(general.dataset_dir, f"experiment_1/T_{tzone}_E_{ym}_nu_{nu}")
            os.makedirs(output_dir, exist_ok=True)

            np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
            np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)

    tzone = float(elastic_simu.t_for_ym_nu)
    nu = elastic_simu.nu_for_ym_t
    for ym in elastic_simu.youngs_modulus:
        ym = float(ym)
        for seed in elastic_simu.seeds:
            name = f"{seed}"
            print(f"\n Running simulation {name} for T={tzone}, E={ym} and nu={nu}")

            u_gt, img_final = create_displacement_image(
                gmsh_mesh=gmsh_cell_shape,
                physical_length=general.physical_length,
                dirichlet=dirichlet,
                t_end=elastic_simu.t_end,
                num_time_steps=elastic_simu.num_time_steps,
                zone_radius=pipette_radius,
                zone_center=tuple(general.pipette_center[:2]),
                traction_zone=tzone,
                youngs_modulus=ym,
                nu=nu,
                eta=elastic_simu.eta,
                n=general.n,
                name=name,
                x_range=tuple(general.x_range),
                y_range=tuple(general.y_range),
                num_points=general.num_points,
                noise_amplitude=general.noise_amplitude,
                num_fourier_modes=general.num_fourier_modes,
                lc=general.lc,
                grain=general.grain,
                seed=seed
            )
            output_dir = os.path.join(general.dataset_dir, f"experiment_2/T_{tzone}_E_{ym}_nu_{nu}")
            os.makedirs(output_dir, exist_ok=True)  

            np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
            np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)
    
    tzone = float(elastic_simu.t_for_ym_nu)
    ym = float(elastic_simu.ym_for_t_nu)
    for nu in elastic_simu.nu:
        for seed in elastic_simu.seeds:
            name = f"{seed}"
            print(f"\n Running simulation {name} for T={tzone}, E={ym} and nu={nu}")

            u_gt, img_final = create_displacement_image(
                gmsh_mesh=gmsh_cell_shape,
                physical_length=general.physical_length,
                dirichlet=dirichlet,
                t_end=elastic_simu.t_end,
                num_time_steps=elastic_simu.num_time_steps,
                zone_radius=pipette_radius,
                zone_center=tuple(general.pipette_center[:2]),
                traction_zone=tzone,
                youngs_modulus=ym,
                nu=nu,
                eta=elastic_simu.eta,
                n=general.n,
                name=name,
                x_range=tuple(general.x_range),
                y_range=tuple(general.y_range),
                num_points=general.num_points,
                noise_amplitude=general.noise_amplitude,
                num_fourier_modes=general.num_fourier_modes,
                lc=general.lc,
                grain=general.grain,
                seed=seed
            )
            output_dir = os.path.join(general.dataset_dir, f"experiment_3/T_{tzone}_E_{ym}_nu_{nu}")
            os.makedirs(output_dir, exist_ok=True)  

            np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
            np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)

if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)