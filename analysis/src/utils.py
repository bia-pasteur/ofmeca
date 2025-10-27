"""Some utils functions"""
#pylint: disable=invalid-name
#pylint: disable=trailing-whitespace
from pathlib import Path
import os
import re
import pandas as pd
import numpy as np 


def extract_E_from_folder(folder_name: str) -> float:
    """
    Extracts the Young's modulus (E) value from a folder name of the format `'T_X_E_Y_nu_Z'`.

    Args:
        folder_name (str):
            Name of the folder containing the encoded parameter values.

    Returns:
        float:
            The extracted Young’s modulus `E`.

    Raises:
        ValueError:
            If the folder name does not contain a valid `_E_` pattern.
    """
    match = re.search(r"_E_(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Could not extract E from {folder_name}")
    return float(match.group(1))


def extract_T_from_folder(folder_name: str) -> float:
    """
    Extracts the applied Traction (T) value from a folder name of the format `'T_X_E_Y_nu_Z'`.

    Args:
        folder_name (str):
            Name of the folder containing the encoded parameter values.

    Returns:
        float:
            The extracted applied Traction `T`.

    Raises:
        ValueError:
            If the folder name does not contain a valid `T_` pattern.
    """
    match = re.search(r"T_(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Could not extract T from {folder_name}")
    return float(match.group(1))


def extract_nu_from_folder(folder_name: str) -> float:
    """
    Extracts the Poisson's ratio (nu) value from a folder name of the format `'T_X_E_Y_nu_Z'`.

    Args:
        folder_name (str):
            Name of the folder containing the encoded parameter values.

    Returns:
        float:
            The extracted Poisson's ratio `nu`.

    Raises:
        ValueError:
            If the folder name does not contain a valid `_nu_` pattern.
    """
    match = re.search(r"_nu_([0-9]*\.?[0-9]+)", folder_name)
    if match is None:
        raise ValueError(f"Could not extract nu from {folder_name}")
    return float(match.group(1))


def find_experiment_folder(base_path: Path, T: float, E: float, nu: float)  -> Path:
    """
    Searches recursively for an experiment folder matching the parameters T, E, and ν.

    The folder structure is expected to follow the convention `'T_X_E_Y_nu_Z'`.

    Args:
        base_path (Path):
            Base directory containing subfolders for different experiments.
        T (float):
            Tension or traction parameter to match.
        E (float):
            Young’s modulus to match.
        nu (float):
            Poisson’s ratio to match.

    Returns:
        Path:
            The path to the matching experiment folder.

    Raises:
        FileNotFoundError:
            If no folder matching the given parameters is found.
    """
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
        for folder in subdir.iterdir():
            if not folder.is_dir():
                continue
            if folder.is_dir():
                t_val = extract_T_from_folder(folder.name)
                e_val = extract_E_from_folder(folder.name)
                nu_val = extract_nu_from_folder(folder.name)
                if np.isclose(t_val, T) and np.isclose(e_val, E) and np.isclose(nu_val, nu):
                    return folder
    raise FileNotFoundError(f"No folder found for T={T}, E={E}")


def extract_std_from_file(file_name: str) -> float:
    """
    Extracts the noise standard deviation value from a file name of the format `'std_0p05_img.npy'`.

    Args:
        file_name (str):
            Name of the file containing the encoded noise level.

    Returns:
        float:
            The extracted noise standard deviation.

    Raises:
        ValueError:
            If the file name does not contain a valid `std_` pattern.
    """
    match = re.search(r"std_(\d+(?:p\d+)?)_img\.npy", file_name)
    if match is None:
        raise ValueError(f"Could not extract std from {file_name}")
    std_str = match.group(1).replace("p", ".")
    return float(std_str)


def get_all_stds_from_folder(folder_path: str) -> list[float]:
    """
    Scans a folder and retrieves all standard deviation values from file names matching the pattern `'std_*_img.npy'`.

    Args:
        folder_path (str):
            Path to the folder containing noisy image files.

    Returns:
        list[float]:
            Sorted list of extracted noise standard deviations.
    """
    stds = []
    for file_name in os.listdir(folder_path):
        if "std_" in file_name and file_name.endswith("_img.npy"):
            std = extract_std_from_file(file_name)
            stds.append(std)
    return sorted(stds)

def load_images_and_displacements(
    exp_folder: Path, 
    mode: str = "original"
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Loads image sequences and corresponding displacement fields from an experiment folder.

    Args:
        exp_folder (str or Path):
            Path to the folder containing `.npy` image and displacement files.
        mode (str, optional):
            Loading mode. Options:
              - `'original'`: loads original images (`*_img.npy`) and displacements (`*_ugt.npy`).
              - `'noisy'`: loads noisy versions (`std_*_img.npy`, `std_*_ugt.npy`).
            Defaults to `'original'`.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            - `images`: List of image arrays.
            - `displacements`: List of displacement field arrays.

    Raises:
        ValueError:
            If `mode` is not `'original'` or `'noisy'`.
    """
    exp_folder = Path(exp_folder)

    if mode == "original":
        img_files = sorted(
            exp_folder.glob("*_img.npy"),
            key=lambda f: int(re.match(r"(\d+)_img\.npy", f.name).group(1)) if re.match(r"(\d+)_img\.npy", f.name) else -1
        )
        disp_files = sorted(
            exp_folder.glob("*_ugt.npy"),
            key=lambda f: int(re.match(r"(\d+)_ugt\.npy", f.name).group(1)) if re.match(r"(\d+)_ugt\.npy", f.name) else -1
        )

    elif mode == "noisy":
        img_files = sorted(
            [f for f in exp_folder.glob("std_*_img.npy")],
            key=lambda f: float(re.search(r"std_(\d+p?\d*)_img\.npy", f.name).group(1).replace("p", "."))
        )
        disp_files = sorted(
            [f for f in exp_folder.glob("std_*_ugt.npy")],
            key=lambda f: float(re.search(r"std_(\d+p?\d*)_ugt\.npy", f.name).group(1).replace("p", "."))
        )
    else:
        raise ValueError("mode must be either 'original' or 'noisy'")

    images = [np.load(f) for f in img_files]
    displacements = [np.load(f) for f in disp_files]

    return images, displacements


def compute_lame(E: float, nu:float) -> tuple[float, float]:
    """
    Computes the Lamé parameters (μ and λ) from Young’s modulus and Poisson’s ratio.

    Args:
        E (float):
            Young’s modulus of the material.
        nu (float):
            Poisson’s ratio of the material.

    Returns:
        tuple[float, float]:
            - `mu_e` (float): Shear modulus (μ).
            - `lambda_e` (float): First Lamé parameter (λ).
    """
    lambda_e = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_e = E / (2 * (1 + nu))
    return mu_e, lambda_e


def rmse(u: np.ndarray, h: np.ndarray) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between two vector fields.

    Args:
        u (np.ndarray):
            Ground-truth field, shape `(d, H, W, ...)`.
        h (np.ndarray):
            Estimated or predicted field, same shape as `u`.

    Returns:
        float:
            The root mean square error value between `u` and `h`.
    """
    diff = u - h
    mse = np.mean(diff ** 2) 
    return np.sqrt(mse)


def results_to_df(results: dict | list[dict]) -> pd.DataFrame:
    """
    Converts optical flow evaluation results into a formatted pandas DataFrame.

    Args:
        results (dict or list[dict]):
            Results containing RMSE values per optical flow method. Each entry should include keys such as:
              - `'mean_rmse_disp'`
              - `'mean_rmse_strain'`
              - `'mean_rmse_def'`
              - `'mean_rmse_stress'`
              - `'mean_rmse_traction'`

    Returns:
        pd.DataFrame:
            A formatted table with RMSE metrics for:
            displacement, strain, deformation, stress, and traction force,
            indexed by method name in the order:
            `["Proposed", "HS", "Farneback", "TV-L1", "ILK"]`.
    """
    name_map = {
        "fista": "Proposed",
        "hs": "HS",
        "farneback": "Farneback",
        "tv_l1": "TV-L1",
        "ilk": "ILK",
    }
    
    desired_order = ["Proposed", "HS", "Farneback", "TV-L1", "ILK"]
    
    if isinstance(results, dict):
        results = [results]

    tables = []
    for res in results:
        if "mean_rmse_disp" in res:
            df = pd.DataFrame({
                "displacement": res["mean_rmse_disp"],
                "strain": res["mean_rmse_strain"],
                "deformation": res["mean_rmse_def"],
                "stress": res["mean_rmse_stress"],
                "traction force": res["mean_rmse_traction"],
            })
        else : 
            df = pd.DataFrame({
                "displacement": res[0]["rmse_flows"],
                "strain": res[0]["rmse_strain"],
                "deformation": res[0]["rmse_def"],
                "stress": res[0]["rmse_stress"],
                "traction force": res[0]["rmse_traction"],
            })
        df.index = [name_map.get(k, k) for k in df.index]

        df = df[["displacement", "strain", "deformation", "stress", "traction force"]]
        tables.append(df)

    df_mean = pd.concat(tables).groupby(level=0).mean().round(4)

    df_mean = df_mean.reindex([m for m in desired_order if m in df_mean.index])

    return df_mean