"""This module is useful for simualtion"""

from typing import List
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from simulation.src.imaging.generator import create_image_simu

def create_displacement_image(
    gmsh_mesh: callable,
    physical_length: float,
    dirichlet: callable,
    t_end: float,
    num_time_steps: int,
    zone_radius: float,
    zone_center: tuple,
    traction_zone: float,
    youngs_modulus: float,
    nu: float,
    eta: float,
    n: float,
    name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_points: int,
    noise_amplitude: float,
    num_fourier_modes: int,
    lc: float,
    grain: float,
    seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic displacement field and corresponding deformed image
    from a micropipette aspiration simulation.

    This function runs a full physical simulation using FEniCSx (through
    `create_image_simu`) to generate a ground-truth displacement field (uGT)
    and a warped image resulting from the deformation of the sample under
    micropipette suction.

    Args:
        gmsh_mesh (callable): Type of Gmsh geometry.
        physical_length (float): Physical length scale of the cell domain in micrometers.
        dirichlet (list[float]): Dirichlet boundary condition definition or displacement constraints.
        t_end (float): Final simulation time.
        num_time_steps (int): Number of time steps in the simulation.
        zone_radius (float): Radius of the active force zone.
        zone_center (tuple): Center coordinates of the force zone.
        traction_zone (float): Applied traction or suction pressure inside the pipette zone.
        youngs_modulus (float): Young’s modulus of the material (elastic stiffness).
        nu (float): Poisson’s ratio of the material.
        eta (float): Viscosity coefficient for viscoelastic materials.
        n (float): Exponent for the viscoelastic or nonlinear viscosity model.
        name (str): Name of the simulation or output image files.
        x_range (tuple[float, float]): Physical range of x-coordinates for image generation.
        y_range (tuple[float, float]): Physical range of y-coordinates for image generation.
        num_points (int): Number of pixels or sampling points per axis.
        noise_amplitude (float): Amplitude of Gaussian noise added to the final image.
        num_fourier_modes (int): Number of Fourier modes for random pattern generation.
        lc (float): Mesh characteristic length for Gmsh generation.
        grain (float): Controls the spatial scale of texture granularity.
        seed (int): Random seed for reproducibility of textures and noise.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - **u_ground_truth** (`np.ndarray`): Displacement field array of shape `(2, T-1, H, W)`,
              representing ground-truth displacements between time steps.
            - **img** (`np.ndarray`): Normalized image (float array in [0, 1])
              representing the deformed sample corresponding to the final state.
    """
    u_list, warped_image_path = create_image_simu(gmsh_mesh, physical_length, dirichlet, t_end, num_time_steps, zone_radius, zone_center, traction_zone, youngs_modulus, nu, eta, n, name, x_range, y_range, num_points=num_points, noise_amplitude=noise_amplitude, num_fourier_modes=num_fourier_modes, lc=lc, grain=grain, seed=seed)
    u_list = u_list.transpose(3, 0, 1, 2)
    u_ground_truth_ = u_list[:,1:]
    u_ground_truth = np.zeros_like(u_ground_truth_)
    u_ground_truth[0] = u_ground_truth_[1]
    u_ground_truth[1] = u_ground_truth_[0]

    img = imread(warped_image_path)
    img = img/img.max()
    return u_ground_truth, img


def create_test_images(
    gmsh_mesh: callable,
    physical_length: float,
    dirichlet: callable,
    t_end: float,
    num_time_steps: int,
    zone_radius: float,
    zone_center: tuple,
    traction_zone: float,
    youngs_modulus: float,
    nu: float,
    eta: float,
    n: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_points: int,
    noise_amplitude: float,
    num_fourier_modes: int,
    lc: float,
    grain: float,
    seeds: list[int]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate multiple synthetic test images and corresponding displacement fields
    for model validation or machine learning training.

    This function repeatedly calls `create_displacement_image()` with
    different random seeds to produce a dataset of images and displacement
    maps under varying physical or random texture conditions.

    Args:
        gmsh_mesh (callable): Type of Gmsh geometry.
        physical_length (float): Physical length of the cell domain.
        dirichlet (callable): Boundary conditions for the simulation.
        t_end (float): Final time for the time-dependent simulation.
        num_time_steps (int): Number of time steps.
        zone_radius (float): Radius of the active force zone.
        zone_center (tuple): Center coordinates of the force zone.
        T_zone (float): Traction or suction pressure applied in the pipette zone.
        E (float): Young’s modulus of the material.
        nu (float): Poisson’s ratio of the material.
        eta (float): Viscosity coefficient.
        n (float): Exponent for the viscoelastic material model.
        x_range (tuple[float, float]): Physical range of x-coordinates for image generation.
        y_range (tuple[float, float]): Physical range of y-coordinates for image generation.
        num_points (int): Number of pixels or sampling points per axis.
        noise_amplitude (float): Noise amplitude for image generation.
        num_fourier_modes (int): Number of Fourier modes for the random texture.
        lc (float): Mesh characteristic length.
        grain (float): Texture granularity parameter.
        seeds (list[int]): Random seeds for reproducibility of each generated image.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            - **images** (`list[np.ndarray]`): List of generated normalized 2D images.
            - **displacements** (`list[np.ndarray]`): List of ground-truth displacement
              fields corresponding to each generated image.
    """
    images = []
    displacements = []

    for i, seed in enumerate(seeds):
        name = f'test_image_{i}'
        u_ground_truth, img = create_displacement_image(gmsh_mesh, physical_length, dirichlet, t_end, num_time_steps, zone_radius, zone_center, traction_zone, youngs_modulus, nu, eta, n, name, x_range, y_range, num_points, noise_amplitude, num_fourier_modes, lc, grain, seed)
        images.append(img)
        displacements.append(u_ground_truth)
    return images, displacements

def create_noisy_images(
    img: np.ndarray,
    disp: np.ndarray,
    stds: List[float],
    seed: int) -> tuple:
    """
    Generate noisy versions of a synthetic image for testing robustness.

    Args:
        img (np.ndarray): 
            Original 2-frame image sequence of shape (2, H, W) with values in [0, 1].
        disp (np.ndarray): 
            Ground-truth displacement field corresponding to `img`.
        stds (List[float]): 
            List of noise standard deviations to apply.
        seed (int): 
            Random seed for reproducibility.

    Returns:
        tuple:
            - noisy_images (list[np.ndarray]): List of noisy images, each of shape (H, W).
            - displacements_list (list[np.ndarray]): List of displacement fields (repeated to match `noisy_images`).
    """
    rng = np.random.default_rng(seed) 
    noisy_images = []

    for std in stds:
        noise = rng.normal(0, std, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)
        noisy_images.append(noisy_img)

    return noisy_images, [disp] * len(stds)

def observe_img(image: np.ndarray) -> None:
    """
    Display the synthetic cell image before and after deformation.

    This function shows two frames of an image sequence side by side:
    the initial undeformed configuration and the deformed configuration
    after applying displacement (e.g., due to micropipette aspiration).

    Args:
        image (np.ndarray): 
            A 3D array of shape (2, H, W) representing the synthetic cell images:
            - image[0]: initial (undeformed) image
            - image[1]: deformed image

    Returns:
        None
    """
    _, axes = plt.subplots(1, 2, figsize=(18, 10)) 

    axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title('Initial synthetic cell image')
    axes[0].axis("off")

    axes[1].imshow(image[1], cmap='gray')
    axes[1].set_title('Deformed synthetic cell image')
    axes[1].axis("off")
    
    plt.show()