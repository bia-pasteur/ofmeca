# Hessian regularized optical flow algorithm for mechanobiology

This project simulates micropipette aspiration experiments on elastic cells and analyzes them with several optical flow algorithms to estimate key mechanical quantities — displacement, strain, deformation, stress, and traction forces.

It accompanies the ISBI 2026 submission:
"A second-order regularized optical flow for mechanical quantification of cellular deformation."

The goal is to evaluate and compare different optical flow methods for estimating cellular mechanics — identifying which one provides the most accurate and physically consistent results under varying noise levels and experimental conditions.

# Overview 

The repository is devided in two principal parts : 

### 1)  Simulation

Generates synthetic images of deformed elastic cells using FEniCSx (finite element simulation).

FEniCSx is computationally heavy. Pre-generated datasets are available at [link to dataset].

### 2) Analysis 

Analyzes the synthetic images using multiple optical flow algorithms, computes the derived mechanical quantities, and compares the results.

It is possible to :

Run the full pipeline (simulate + analyze), or

Run only one part (e.g., use pre-generated data for analysis).


# Installation and full pipeline execution 

Start by cloning the repository

`git clone git@github.com:jlahmani/ofmeca.git`

Then create a conda environment with python=3.12 and install fenicsx using 

`conda install -c conda-forge fenics-dolfinx`

Install requirements

`pip install -r requirements.txt`

Run the code to create the images and perform the analysis using 

`./run_all.sh`


# Creation of synthetic images

First create a conda environment with python=3.12 and install fenicsx using 

`conda install -c conda-forge fenics-dolfinx`

Install requirements

`pip install -r simulation/requirements.txt`

## Original images

There are three simulation experiments:

| Experiment | Variable | Fixed parameters | YAML keys |
|-------------|-----------|------------------|------------|
| 1 | Traction `T` | E, ν | `traction_zone`, `ym_for_t_nu`, `nu_for_t_ym` |
| 2 | Young’s modulus `E` | T, ν | `youngs_modulus`, `t_for_ym_nu`, `nu_for_t_ym` |
| 3 | Poisson’s ratio `ν` | T, E | `nu`, `t_for_ym_nu`, `ym_for_t_nu` |

Each setting is repeated for all `seeds` (random initial shapes).

Run the code to create these images using 

`./run_simulation_original.sh`

Images and associated displacement are saved under:

`data/experiment_i/T_<T>_E_<E>_nu_<nu>/`

## Noisy images

To test the robustness to noise, Gaussian noise of increasing stds (defined by `noise_stds`) is added to all reference images determined by `T_for_noisy_exp`, `E_for_noisy_exp`, `nu_for_noisy_exp`.

Run the code to create these images using 

`./run_simulation_noise.sh`

Images and associated displacements are saved under:

`data/noise_experiment_T_<T>_E_<E>_nu_<nu>/img_<seed>)`

## Note
Other geometric and imaging parameters can be tuned in: `simulation/configs/simulations.yaml`

# Optical flow analysis and mechanical quantification

Install requirements 

`pip install -r analysis/requirements.txt`

### Edit: `analysis/configs/experiment.yaml`.

- `of_funcs`: list of optical flow algorithms to test.

- Select images by:

    -  Explicit IDs: (`T`, `E`, `nu`, `image_id`)

    - Whole experiment: (`exp_ind`)

- Or run all experiments by leaving these unspecified.

Results (RMSE tables, plots, etc.) are stored under `results/tables/` and `results/plots/`

### Plotting controls:

In `experiments.yaml`, the section plot_parameters defines which images to visualize:

`T_for_plot`, `E_for_plot`, `nu_for_plot`, `implot`

### Regularization study:

Use `image_for_test_reg`, `T_for_reg`, `E_for_reg`, `nu_for_reg`, and `factors_for_reg` to control the regularization experiments.

### Running the scripts:

| Purpose | Command | Output |
|-------------|-----------|------------|
| Run experiments on regular images | `./run_exp.sh` | RMSE table + plots |
| Run regularization robustness study | `./run_noise_exp.sh` | Combined plot (no saved tables) |
| Run noise robustness study | `./run_reg_exp.sh` | Saved results + plots |
| Run regularization and noise robustness study | `./run_reg_noise_exp.sh` | Saved results + plots |