"""Useful to create noisy images from existing synthetic images"""
import os
import numpy as np
import jsonargparse
from data_generation.src.config import NoiseSimuParams
from data_generation.src.simulation_pipeline import create_noisy_images

def main(
    noise_simu: NoiseSimuParams
):
    """
    Generates noisy versions of existing simulated images and associate the real displacement fields.

    This function locates the original simulation dataset corresponding to a specific 
    traction force (T), Young’s modulus (E), and Poisson’s ratio (ν), and generates 
    noisy image sequences according to the specified noise levels. The resulting noisy 
    images and displacement fields are saved in structured directories.

    Args:
        exp (ExperimentsParams): Experiment parameters

    Raises:
        FileNotFoundError:
            If the original dataset corresponding to the requested T, E, and ν is not found.

    Notes:
        Output files are saved in directories following the convention `data/noise_experiment_T_{T}_E_{E}_nu_{nu}/img_{seed}/`.
        For each noise level, images and displacements are saved as `std_<std>_img.npy` and `std_<std>_ugt.npy`.
    """
    dataset_root = "data"
    noise_output_root = f"data/noise_experiment_T_{noise_simu.traction_zone}_E_{noise_simu.ym}_nu_{noise_simu.nu}"
    os.makedirs(noise_output_root, exist_ok=True)
    
    exp_candidates = ["experiment_1", "experiment_2", "experiment_3"]
    input_dir = None
    for exp_name in exp_candidates:
        candidate_dir = os.path.join(dataset_root, f"{exp_name}/T_{noise_simu.traction_zone}_E_{noise_simu.ym}_nu_{noise_simu.nu}")
        if os.path.exists(candidate_dir):
            input_dir = candidate_dir
            break
    if input_dir is None:
        raise FileNotFoundError(f"Requested T={noise_simu.traction_zone}, E={noise_simu.ym}, nu={noise_simu.nu} not found in any experiment.")
    
    for seed in noise_simu.seeds:
        img_path = os.path.join(input_dir, f"{seed}_img.npy")
        disp_path = os.path.join(input_dir, f"{seed}_ugt.npy")
        img = np.load(img_path)
        disp = np.load(disp_path)
        seed_output_dir = os.path.join(noise_output_root, f"img_{seed}")
        os.makedirs(seed_output_dir, exist_ok=True)
        
        np.save(os.path.join(seed_output_dir, "std_0_img.npy"), img)
        np.save(os.path.join(seed_output_dir, "std_0_ugt.npy"), disp)
        print("Running noise simulation")
        noisy_images, disps = create_noisy_images(img, disp, noise_simu.noise_stds, seed)
        print("Noisy images done !")
        for std, noisy_img, disp_gt in zip(noise_simu.noise_stds, noisy_images, disps):
            std_tag = f"{std:.2f}".replace(".", "p")
            np.save(os.path.join(seed_output_dir, f"std_{std_tag}_img.npy"), noisy_img)
            np.save(os.path.join(seed_output_dir, f"std_{std_tag}_ugt.npy"), disp_gt)

if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)