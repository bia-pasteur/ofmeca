from pathlib import Path
from types import SimpleNamespace
import yaml
import numpy as np
import dolfinx.fem as fem
from dolfinx import default_scalar_type
import matplotlib.pyplot as plt
from mechanics.src.MCM.quantities_computation import strain_mask, deformation
from mechanics.src.optical_flow.algorithms import fista_of, hs_of
from mechanics.src.VE.inverse_pb import get_E, get_eta
from data_generation.src.mesh.creation import create_mesh_file, gmsh_cell_shape
from typing import Tuple
from petsc4py.PETSc import ScalarType # pylint: disable=no-name-in-module
import numpy as np 
from dolfinx import default_scalar_type, plot
import dolfinx.fem as fem
import dolfinx.mesh as mesh 
import ufl
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, apply_lifting, set_bc
from petsc4py import PETSc
from ufl import TrialFunction, TestFunction, inner, grad, adjoint, derivative, action, Measure
from mechanics.src.utils import compute_lame
from data_generation.src.fem.solver import create_dirichlet, strain, stress_elas, stress_visc
from scipy.optimize import minimize
from data_generation.src.imaging.generator import dirichlet, deform_mesh
import pyvista
from petsc4py import PETSc
import time
from mechanics.src.utils import rmse


def run_VE_maps(
    ym, 
    eta, 
    t_zone, 
    seed, 
    physical_length, 
    num_points, 
    noise_amplitude, 
    num_fourier_modes, 
    lc, 
    zone_radius, 
    zone_center, 
    x_range, 
    y_range, 
    t_end, 
    num_time_steps, 
    ym_init, 
    alpha_ym,
    bounds_ym, 
    iter_ym,
    ftol_ym,
    gtol_ym, 
    eta_init, 
    alpha_eta,
    nb_steps,
    bounds_eta, 
    iter_eta,
    ftol_eta,
    gtol_eta, 
    ):

    ### Load images and displacement
    start_time = time.time()
    folder = Path(f'/Users/josephinelahmani/Desktop/ofmeca/data/viscoelas/T_{t_zone}_E_{ym}_nu_0.3_eta_{eta}_patches')
    img_path = folder / f'{seed}_img.npy'
    ugt_path = folder / f'{seed}_ugt.npy'
    img = np.load(img_path)
    dis = np.load(ugt_path)
    CONFIG_PATH = "/Users/josephinelahmani/Desktop/ofmeca/mechanics/configs/analysis.yaml"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    PARAMS = SimpleNamespace(**cfg["optical_flow"]["fista"])

    dis_of_ = fista_of(img[:10], PARAMS, True)
    mask = (dis[0,0] != 0)
    dis_of = (dis_of_[:,1:] / PIXEL_SIZE) * mask
    
    ### Mesh creation

    x_c, y_c, msh = create_mesh_file(physical_length, gmsh_cell_shape, np.random.default_rng(seed=seed), num_points, noise_amplitude, num_fourier_modes, lc)

    ### Computation of E
    
    r_ym = get_E(ym_init, dis_of[:,-2:-1], x_range, y_range, msh, t_zone, alpha_ym, zone_radius, zone_center, x_c, y_c, physical_length, bounds_ym, iter_ym, ftol_ym, gtol_ym)

    print('Time for E:', time.time() - start_time)
    
    ### Computation of eta
    
    r_eta = get_eta(eta_init, r_ym, nb_steps, dis_of, x_range, y_range, msh, t_zone, alpha_eta, zone_radius, zone_center, x_c, y_c, physical_length, t_end, num_time_steps, bounds_eta, iter_eta, ftol_eta, gtol_eta)
    ### Extreme values + optional vizualisation

    Q_ = fem.functionspace(msh, ("CG", 1))
    E_ = fem.Function(Q_)
    E_.x.array[:] = r_ym.x
    eta_ = fem.Function(Q_)
    eta_.x.array[:] = r_eta.x

    print("computed E min/max/moy:", E_.x.array.min(), E_.x.array.max(), E_.x.array.mean())
    print("computed eta min/max/moy:", eta_.x.array.min(), eta_.x.array.max(), eta_.x.array.mean())

    points_ym = msh.geometry.x
    cells_ym = msh.topology.connectivity(2, 0).array.reshape(-1, 3)

    plt.figure()
    plt.tripcolor(points_ym[:, 0], points_ym[:, 1], cells_ym, facecolors=E_.x.array[cells_ym].mean(axis=1),
                shading='flat', cmap="viridis")
    plt.colorbar()
    plt.title("E")
    plt.show()
    
    points_eta = msh.geometry.x
    cells_eta = msh.topology.connectivity(2, 0).array.reshape(-1, 3)

    plt.figure()
    plt.tripcolor(points_eta[:, 0], points_eta[:, 1], cells_eta, facecolors=eta_.x.array[cells_eta].mean(axis=1),
                shading='flat', cmap="viridis")
    plt.colorbar()
    plt.title("eta")
    plt.show()
    
    print('Execution time:', time.time() - start_time)
    

PHYSICAL_LENGTH = 1.0
NUM_POINTS = 100
NOISE_AMPLITUDE = 0.2
NUM_FOURIER_MODES = 7
LC = 0.5
T_ZONE = 100
ZONE_FACTOR = 0.5
ZONE_RADIUS = ZONE_FACTOR * PHYSICAL_LENGTH
ZONE_CENTER = [0, 1, 0]
PIXEL_SIZE = 300 / (4 * PHYSICAL_LENGTH) # for computed optical flow
X_RANGE = [-2, 2]
T_END = 5
NUM_TIME_STEPS = 40
YM_INIT = 1000.0
NB_STEPS = 8
ETA_INIT = 200.0

ALPHA_YM = 1e-9
BOUNDS_YM =[(0, 20000)]
ITER_YM = 50
FTOL_YM = 0
GTOL_YM = 1e-11
ALPHA_ETA = 3e-9
BOUNDS_ETA = [(0, 4000)]
ITER_ETA = 50
FTOL_ETA = 0
GTOL_ETA = 1e-11

E = 1000.0
ETA = 150.0
T_ZONE = 100.0

# for img_seed in [1]:
#     print('E:', E, 'eta:', ETA, 'image:', img_seed)
#     run_VE_maps(E, ETA, T_ZONE, img_seed, PHYSICAL_LENGTH, NUM_POINTS, NOISE_AMPLITUDE,
#             NUM_FOURIER_MODES, LC, ZONE_RADIUS, ZONE_CENTER, X_RANGE, X_RANGE, T_END,
#             NUM_TIME_STEPS, YM_INIT, ALPHA_YM, BOUNDS_YM, ITER_YM, FTOL_YM, GTOL_YM, ETA_INIT,
#             ALPHA_ETA, NB_STEPS, BOUNDS_ETA, ITER_ETA, FTOL_ETA, GTOL_ETA)

folder = Path(f'/Users/josephinelahmani/Desktop/ofmeca/data/viscoelas/T_{T_ZONE}_E_{E}_nu_0.3_eta_{ETA}_patches')
img_path = folder / '1_img.npy'
ugt_path = folder / '1_ugt.npy'
img = np.load(img_path)
dis = np.load(ugt_path)

# fig, axes = plt.subplots(1, 2, figsize=(15, 4))

# im0 = axes[0].imshow(img[0])
# axes[0].set_title("img init")
# plt.colorbar(im0, ax=axes[0])

# im1 = axes[1].imshow(img[-1])
# axes[1].set_title("img final")
# plt.colorbar(im1, ax=axes[1])

# plt.tight_layout()
# plt.show()

CONFIG_PATH = "/Users/josephinelahmani/Desktop/ofmeca/mechanics/configs/analysis.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
PARAMS = SimpleNamespace(**cfg["optical_flow"]["fista"])

dis_of_ = fista_of(img[:15], PARAMS, True)
mask = (dis[0,0] != 0)
dis_of = dis_of_[:,1:] * mask

dis = dis * PIXEL_SIZE

strain_gt = strain_mask(dis, [1, 1], mask)
strain_of = strain_mask(dis_of, [1, 1], mask)

for im in range(dis_of.shape[1]):
    rmse_flow = rmse(dis_of[:,im], dis[:,im])
    rmse_strain = rmse(strain_gt[:,:,im], strain_of[:,:,im])
    
    print('image', im, 'rmse flow', rmse_flow, 'rmse_strain', rmse_strain)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(dis[0,0])
axes[0].set_title("dis[0,0]")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(dis_of[0,0])
axes[1].set_title("dis_of[0,0]")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(dis[0,0] - dis_of[0,0])
axes[2].set_title("dis[0,0]-dis_of[0,0]")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(dis[0,-1])
axes[0].set_title("dis[0,-1]")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(dis_of[0,-1])
axes[1].set_title("dis_of[0,-1]")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()

def_gt = deformation(strain_gt)
def_of = deformation(strain_of)

print(def_gt.shape)
print(def_of.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(strain_gt[0,0,-1], vmin=0)
axes[0].set_title("norm strain gt")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(strain_of[0,0,-1])
axes[1].set_title("norm strain of")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()