"""Generator of the images"""

# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from typing import Callable, Union, List, Tuple, Optional
import os
import numpy as np
import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.mesh import compute_midpoints
from petsc4py.PETSc import ScalarType # pylint: disable=no-name-in-module
import tifffile
from noise import pnoise2
from simulation.src.mesh.creation import create_mesh_file
from simulation.src.fem.solver import finite_elements_force_zone


def closest_cell(
    mesh: dolfinx.mesh.Mesh,
    point: np.ndarray,
    candidate_cells: np.ndarray
) -> int:
    """
    Find the cell (element) in the mesh that is closest to a given point,
    restricted to a list of candidate cells.

    Args:
        mesh (dolfinx.mesh.Mesh): Computational mesh.
        point (np.ndarray): Target point in physical coordinates, shape (3,).
        candidate_cells (np.ndarray): Array of candidate cell indices.

    Returns:
        int: Index of the cell in `candidate_cells` whose midpoint
             is closest to the given point.
    """
    midpoints = compute_midpoints(mesh, mesh.topology.dim, np.array(candidate_cells, dtype=np.int32))
    dists = np.linalg.norm(midpoints - point, axis=1)
    return candidate_cells[np.argmin(dists)]


def interpolation(
    mesh: dolfinx.mesh.Mesh,
    u: fem.Function,
    n: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Interpolates a finite element vector field `u` on a regular 2D or 3D grid.

    Args:
        mesh (dolfinx.mesh.Mesh): Computational mesh where `u` is defined.
        u (dolfinx.fem.Function): Vector-valued FE function (displacement field).
        n (int): Number of grid points along each axis of the interpolation grid.
        x_range (tuple): (xmin, xmax) bounds of interpolation domain along x-axis.
        y_range (tuple): (ymin, ymax) bounds of interpolation domain along y-axis.
        z_range (tuple, optional): (zmin, zmax) bounds along z-axis. If None → 2D interpolation.

    Returns:
        np.ndarray: Array of interpolated values.
            - Shape (n, n, 2) for 2D interpolation
            - Shape (n, n, n, 3) for 3D interpolation
    """
    xmin, xmax = x_range
    ymax, ymin = y_range
    
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymax, ymin, n)
    
    if z_range is None:
        # 2D case
        X, Y = np.meshgrid(x, y, indexing='xy')
        #X = np.flip(X, axis=0)
        #Y = np.flip(Y)
        pixels = np.stack((X, Y, np.zeros_like(X)), axis=-1)
        dim = 2
        val_shape = (n, n, 2)
    else:
        # 3D case
        zmin, zmax = z_range
        z = np.linspace(zmin, zmax, n)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pixels = np.stack((X, Y, Z), axis=-1)
        dim = 3
        val_shape = (n, n, n, 3)

    flattened_pix = pixels.reshape(-1, 3)
    
    # Build bounding box tree for the correct dimension
    bounding_box = dolfinx.geometry.bb_tree(mesh, dim)
    
    # Find colliding cells
    cells = dolfinx.geometry.compute_collisions_points(bounding_box, flattened_pix)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cells, flattened_pix)

    pixels_interpolation = []
    cells_interpolation = []

    for i, pixel in enumerate(flattened_pix):
        if len(colliding_cells.links(i)) > 0:
            best_cell = closest_cell(mesh, pixel, colliding_cells.links(i))
            pixels_interpolation.append(pixel)
            cells_interpolation.append(best_cell)

    pixels_interpolation = np.array(pixels_interpolation, dtype=np.float64)
    
    u_values = u.eval(pixels_interpolation, cells_interpolation)

    # Fill the output array
    val = np.zeros(val_shape)
    count = 0

    if dim == 2:
        for i in range(n):
            for j in range(n):
                if count < len(pixels_interpolation):
                    if np.array_equal(pixels[i, j], pixels_interpolation[count]):
                        val[i, j, :] = u_values[count, :2]
                        count += 1
    else:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if count < len(pixels_interpolation):
                        if np.array_equal(pixels[i, j, k], pixels_interpolation[count]):
                            val[i, j, k, :] = u_values[count, :]
                            count += 1

    return val


def dirichlet(
    nodes: np.ndarray, 
    pipette_center: tuple, 
    pipette_radius: float, 
    boundary_x: np.ndarray = None, 
    boundary_y: np.ndarray = None,
    physical_length: float = None) -> np.ndarray:
    """
    Determines if nodes are on the boundary (matching x_coords, y_coords)
    and outside the pipette zone in y-direction.
    
    Args:
        nodes (np.ndarray): Coordinates of points on facets, shape (d, N)
        pipette_center (tuple): Coordinates of the pipette center (x0, y0, z0)
        pipette_radius (float): Radius of the pipette
        boundary_x (np.ndarray): x-coordinates of the boundary points
        boundary_y (np.ndarray): y-coordinates of the boundary points
        eps (float): tolerance for matching boundary points

    Returns:
        np.ndarray: Boolean array of length N, True means Dirichlet should apply
    """
    x, y, z = nodes
    
    if np.all(z == 0) and (boundary_x is None or boundary_y is None):
        raise ValueError("boundary_x and boundary_y cannot be None in 2D")
    n = x.shape[0]

    if boundary_x is not None:
        # Check if (x, y) is close to any boundary point
        on_boundary = np.zeros(n, dtype=bool)
        for bx, by in zip(boundary_x, boundary_y):
            on_boundary |= (np.isclose(x, bx) & np.isclose(y, by))

        # Exclude pipette region along y
        in_pipette = (y > 0) & (x >= -pipette_radius) & (x <= pipette_radius)

        # Dirichlet applies only to boundary points outside pipette
        return np.logical_and(on_boundary, ~in_pipette)
    x, y, z = nodes
    x_zone, y_zone, z_zone = pipette_center
    on_surface = np.isclose(x**2 + y**2 + z**2, physical_length**2)
    distance_to_zone_center = np.sqrt((x-x_zone)**2 + (y - y_zone)**2 + (z-z_zone)**2)
    out_of_zone = distance_to_zone_center >= 2*pipette_radius
    return np.logical_and(on_surface, out_of_zone)


def create_intensities_perlin(mesh: dolfinx.mesh.Mesh, scale: float = 5.0, grain: int = 3, seed: int = 0) -> dolfinx.fem.Function:
    """
    Create a discontinuous Galerkin (DG) function on the mesh where
    the cell-wise values follow a 2D Perlin noise pattern.

    This function assigns a Perlin noise intensity value to each cell of the mesh,
    normalized between 0 and 1, and stores it in a DG finite element function.

    Args:
        mesh (dolfinx.mesh.Mesh): The input computational mesh.
        scale (float): Frequency of the Perlin noise. Smaller values
            produce larger-scale (smoother) patterns.
        grain (int): Polynomial degree of the DG elements.
        seed (int): Random seed for the Perlin noise generator.

    Returns:
        dolfinx.fem.Function: DG function representing the normalized
            Perlin noise intensity field over the mesh.
    """
    Q = fem.functionspace(mesh, ("DG", grain))
    intensities = fem.Function(Q)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    coords = compute_midpoints(mesh, mesh.topology.dim, np.arange(num_cells))

    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    coords_norm = np.zeros_like(coords)
    coords_norm[:,0] = (coords[:,0]-xmin)/(xmax-xmin)
    coords_norm[:,1] = (coords[:,1]-ymin)/(ymax-ymin)

    values_per_cell = np.zeros(num_cells)
    for i in range(num_cells):
        x = coords_norm[i,0] * scale
        y = coords_norm[i,1] * scale
        values_per_cell[i] = pnoise2(x, y, octaves=5, persistence=0.5, repeatx=1024, repeaty=1024, base=seed)
    min_val, max_val = np.percentile(values_per_cell, [1, 99])  
    values_per_cell = np.clip(values_per_cell, min_val, max_val)
    values_per_cell = (values_per_cell - min_val) / (max_val - min_val)

    dofmap = Q.dofmap
    intensities_vals = np.zeros_like(intensities.x.array)
    for cell_index in range(num_cells):
        dofs = dofmap.cell_dofs(cell_index)
        intensities_vals[dofs] = values_per_cell[cell_index]

    intensities.x.array[:] = intensities_vals
    return intensities


def deform_mesh(mesh: dolfinx.mesh.Mesh, u: dolfinx.fem.Function) -> dolfinx.mesh.Mesh:
    """
    Deform the input mesh according to a given displacement field.

    The displacement field `u` is evaluated at the mesh geometry points,
    and each node of the mesh is updated by adding the corresponding
    displacement vector.

    Args:
        mesh (dolfinx.mesh.Mesh): The computational mesh to deform.
        u (dolfinx.fem.Function): Displacement field defined on the mesh.

    Returns:
        dolfinx.mesh.Mesh: The same mesh object with updated coordinates.
    """
    x = mesh.geometry.x
    # Initialise cell search
    bounding_box = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bounding_box, x)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x)
    for i, point in enumerate(x):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    # Evaluate u
    u_values = u.eval(points_on_proc, cells)
    # Update mesh coordinates
    x += u_values

    return mesh
  
  
def create_image_simu(
    mesh_function: Callable,
    physical_length: float,
    dirichlet_: Callable,
    t_end: float,
    num_time_steps: int,
    zone_radius: float,
    zone_center: tuple,
    traction_zone: float,
    youngs_modulus: Union[float, List[float]],
    nu: Union[float, List[float]],
    eta: Union[float, List[float]],
    n: int,
    name: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Optional[Tuple[float, float]] = None,
    regions: Optional[List[Tuple[float, float]]] = None,
    num_points: Optional[float] = None,
    noise_amplitude: Optional[float] = None,
    num_fourier_modes: Optional[float] = None,
    lc=None,
    seed=1, 
    grain=2
) -> Tuple[np.ndarray, str]:
    """
    Generate a sequence of simulated microscopy images based on a
    finite element (FE) deformation simulation with Perlin noise texture.

    The simulation first creates a mesh and a Perlin noise intensity field,
    then applies a time-dependent displacement due to a local force (defined
    by `zone_center` and `zone_radius`), and finally interpolates the deformed
    images on a regular grid to produce synthetic 2D or 3D image sequences.

    Args:
        mesh_function (Callable): Function used to generate the mesh geometry.
        physical_length (float): Physical size of the simulated domain.
        dirichlet_ (Callable): Function defining Dirichlet boundary conditions.
        t_end (float): Final simulation time.
        num_time_steps (int): Number of time steps in the simulation.
        zone_radius (float): Radius of the active force zone.
        zone_center (tuple): Center coordinates of the force zone.
        traction_zone (float): Magnitude of the applied traction.
        youngs_modulus (float or List[float]): Elastic modulus (possibly region-dependent).
        nu (float or List[float]): Poisson ratio.
        eta (float or List[float]): Viscosity coefficient(s).
        n (int): Number of grid points for interpolation.
        name (str): Base name of the output TIFF file.
        x_range (Tuple[float, float]): Spatial range in the x-direction.
        y_range (Tuple[float, float]): Spatial range in the y-direction.
        z_range (Optional[Tuple[float, float]]): Spatial range in z (for 3D).
        regions (Optional[List[Tuple[float, float]]]): List of radial region definitions.
        num_points (Optional[float]): Number of mesh points (if generated procedurally).
        noise_amplitude (Optional[float]): Amplitude of the geometric noise on the mesh.
        num_fourier_modes (Optional[float]): Number of Fourier modes for shape perturbation.
        lc (Optional[float]): Characteristic length of mesh elements.
        seed (int): Random seed.
        grain (int): Degree of the DG function used for Perlin intensities.

    Returns:
        Tuple[np.ndarray, str]:
            - U_list: Array of displacement fields interpolated on a regular grid,
              shape (num_time_steps, n, n, 2) in 2D or (num_time_steps, n, n, n, 3) in 3D.
            - warped_image_path: Path to the saved multi-dimensional TIFF file.
    """

    rng = np.random.default_rng(seed=seed)
    # Create the mesh
    x_coords, y_coords, msh = create_mesh_file(physical_length, mesh_function, rng, num_points, noise_amplitude, num_fourier_modes, lc)
    # Create the function space
    # Lagrange elements, degree 1 (linear elements), vector-valued function space (one function per spatial dimension (3))
    V = fem.functionspace(msh, ('Lagrange', 2, (msh.geometry.dim,)))
    traction_constant = fem.Constant(msh, default_scalar_type((0, traction_zone, 0)))
    
    # Definition of the initial condition 
    # We initialize everything at 0
    def cond_init(x):
        return np.array([0.0*x[0], 0.0*x[1], 0.0*x[2]], dtype=ScalarType)
    
    # Definition of the intensities on the mesh
    intensities = create_intensities_perlin(msh, seed=seed, grain=grain)
    
    # Create original image
    original_image = interpolation(msh, intensities, n, x_range, y_range, z_range)[:,:,0]
  
    # Get displacement when the force is applied on the zone
    dis = finite_elements_force_zone(msh, V, cond_init, dirichlet_, t_end, num_time_steps, zone_radius, zone_center, physical_length, traction_constant, youngs_modulus, nu, eta, x_coords, y_coords, regions)
    if z_range is None:
        N, M = original_image.shape
        
        warped_images = np.zeros((num_time_steps, N, M), dtype=np.float32)
        u_list = np.zeros((num_time_steps, n, n, 2), dtype=np.float32)

        for time in range(num_time_steps):
            uh = dis[time]
            u_list[time] = interpolation(msh, uh, n, x_range, y_range, z_range)
            msh = deform_mesh(msh, uh)
            warped_images[time] = interpolation(msh, intensities, n, x_range, y_range, z_range)[:,:,0]
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data/raw")
        output_dir = os.path.join(base_dir, f"T_{traction_zone}_E_{youngs_modulus}")
        os.makedirs(output_dir, exist_ok=True)
        warped_image_path = os.path.join(output_dir, f"{name}.tiff")
        tifffile.imwrite(warped_image_path, warped_images, metadata={'axes': 'TYX'}, imagej=True)
    return u_list, warped_image_path