"""Solves the FEM problem"""

# pylint: disable=invalid-name
from typing import Callable, List, Union
import ufl
import numpy as np
import dolfinx
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, apply_lifting, set_bc
from ufl import inner, TrialFunction, TestFunction
from petsc4py import PETSc

def strain(u: ufl.Argument) -> ufl.Form:
    """
    Computes the strain of the displacement field

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
    Returns:
        ufl.Form: The strain tensor
            Shape: (d, d)
    """
    return ufl.sym(ufl.grad(u))


def stress_elas(
    u: ufl.Argument, 
    lambda_e: float, 
    mu_e: float
    ) -> ufl.Form:
    """
    Computes the elastic stress of the displacement field (Hooke's law)

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
        lambda_e (float): Lamé elastic parameter
        mu_e (float): Lamé elastic parameter

    Returns:
        ufl.Form: The elastic stress tensor
            Shape: (d, d)
    """
    return lambda_e * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu_e * strain(u)


def stress_visc(
    u: ufl.Argument, 
    lambda_v: float, 
    mu_v: float, 
    delta_t: float
    ) -> ufl.Form:
    """
    Computes the viscous stress tensor of the displacmeent field

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
        lambda_v (float): Lamé viscous parameter
        mu_v (float): Lamé viscous parameter
        delta_t (float): The time step 

    Returns:
        ufl.Form: The viscous stress tensor
            Shape: (d, d)
    """
    return (lambda_v/delta_t)*ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * (mu_v/delta_t) * strain(u)


def create_dirichlet(
    msh: mesh.Mesh, 
    dfacets: callable, 
    V: fem.FunctionSpace, 
    pipette_radius: float, 
    pipette_center: tuple, 
    boundary_x: np.ndarray = None, 
    boundary_y: np.ndarray = None, 
    physical_length: float = None
    ) -> fem.DirichletBC:
    """
    Creates a Dirichlet BC on the mesh boundary outside the pipette region.
    """
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim,
        lambda xx: dfacets(xx, pipette_center, pipette_radius, boundary_x, boundary_y, physical_length)
    )

    u_dirichlet = np.array([0, 0, 0], dtype=default_scalar_type)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    return fem.dirichletbc(u_dirichlet, dofs, V)


def finite_elements_force_zone(    
    msh: dolfinx.mesh.Mesh,
    V: fem.FunctionSpace,
    cond_init: Callable[[np.ndarray], np.ndarray],
    dirichlet: Callable[[np.ndarray, tuple, float, float], np.ndarray],
    t_end: float,
    num_time_steps: int,
    zone_radius: float,
    zone_center: tuple,
    physical_length: float,
    traction_constant: fem.Function,
    youngs_modulus: Union[float, List[float]],
    nu: Union[float, List[float]],
    eta: Union[float, List[float]],
    x_coords = None,
    y_coords = None,
    regions: List[float] = None
) -> List[np.ndarray]:
    """
    Performs a micropipette aspiration simulation with linear viscoelasticity, with optional varying material parameters.

    Args:
        msh (mesh.Mesh): Finite element mesh of the computational domain
        V (fem.FunctionSpace): Function space for displacements
        cond_init (Callable): Initial condition function (displacement field)
        dirichlet (Callable): Function defining Dirichlet BCs at the pipette
        t_end (float): Total simulation time
        num_time_steps (int): Number of discrete time steps
        zone_radius (float): Radius of the active force zone.
        zone_center (tuple): Center coordinates of the force zone.
        physical_length (float): Characteristic physical length (scaling parameter)
        traction_constant (fem.Function): Traction force applied on the pipette boundary
        youngs_modulu (float | list): Young’s modulus (scalar or per-region list)
        nu (float | list): Poisson’s ratio (scalar or per-region list)
        eta (float | list): Viscosity (scalar or per-region list)
        regions (list, optional): Radial subregions [(rmin, rmax), ...]
                                  for spatially varying material parameters

    Returns:
        List[fem.Function]: Displacement field at each time step
    """
    if msh.topology.dim == 2 and (x_coords is None or y_coords is None):
        raise ValueError("x_coords and y_coords cannot be None in 2D")
    # Define the time step
    delta_t = t_end/num_time_steps
    # Compute Lamé parameters
    def lame(E__, nu__, eta__):
        # Elastic Lamé parameters
        lambda_e = E__*nu__ / ((1+nu__)*(1-2*nu__))
        mu_e = E__ / (2*(1+nu__))

        # Viscous Lamé parameters
        lambda_v = -eta__/3
        mu_v = eta__/2
        return lambda_e, mu_e, lambda_v, mu_v
    
    if regions is not None:
        F = fem.functionspace(msh, ("DG", 0))
        youngs_modulus_ = fem.Function(F)
        eta_ = fem.Function(F)
        nu_ = fem.Function(F)
        for i, region in enumerate(regions):
            rmin = region[0]
            rmax = region[1]
            def omega_i(x, rmin_=rmin, rmax_=rmax):
                norms = x[0]**2 + x[1]**2 + x[2]**2
                return np.logical_and(norms >= rmin_**2 - 0.2, norms <= rmax_**2 + 0.2)
            dim = msh.topology.dim
            cells_i = dolfinx.mesh.locate_entities(msh, dim, omega_i)
            youngs_modulus_.x.array[cells_i] = np.full_like(cells_i, youngs_modulus[i], dtype=default_scalar_type)
            eta_.x.array[cells_i] = np.full_like(cells_i, eta[i], dtype=default_scalar_type)
            nu_.x.array[cells_i] = np.full_like(cells_i, nu, dtype=default_scalar_type)
    else : 
        youngs_modulus_ = youngs_modulus
        nu_ = nu
        eta_ = eta
    lambda_e, mu_e, lambda_v, mu_v = lame(youngs_modulus_, nu_, eta_)
    
    # Define a function for the old values
    u_old = fem.Function(V, name="Previous step")
    u_old.interpolate(cond_init)

    # Define solution variable, and interpolate initial solution
    uh = fem.Function(V, name="Solution variable")
    uh.interpolate(cond_init)
    
    # Definition of the integration measures
    ds = ufl.Measure("ds", domain=msh)
    dx = ufl.Measure("dx", domain=msh)
    displacement_history = []
    
    # Create dirichlet conditions 
    bc = create_dirichlet(msh, dirichlet, V, zone_radius, zone_center, x_coords, y_coords, physical_length)
    
    # Create the trial and test function
    u, v = TrialFunction(V), TestFunction(V)

    # Weak form
    a = inner(stress_elas(u, lambda_e, mu_e) + stress_visc(u, lambda_v, mu_v, delta_t), strain(v))*dx
    L = inner(traction_constant, v)*ds + inner(stress_visc(u_old, lambda_v, mu_v, delta_t), strain(v))*dx

    # Convert the left hand side (LSH) and the right hand side (RHS) into a DOLFINx-compatible representation
    compiled_a = fem.form(a)
    compiled_L = fem.form(L)

    # Allocate memory for the solution
    b = fem.Function(V)

    # Assemble the stiffness matrix A from a and apply Dirichlet boundary conditions
    # Linear system : Au = b
    A = assemble_matrix(compiled_a, bcs=[bc])
    A.assemble()

    # Create the solver 
    solver = PETSc.KSP().create(msh.comm) # pylint: disable=no-member

    # Set the system matrix
    solver.setOperators(A)

    # Using direct factorization
    solver.setType(PETSc.KSP.Type.PREONLY) # pylint: disable=no-member

    # Using LU decomposition as a precondition
    # Fast for small to medium problems, memory-intensive
    solver.getPC().setType(PETSc.PC.Type.LU) # pylint: disable=no-member
    
    displacement_history.append(uh.copy())
    
    # Iteratively solve the problem
    for _ in range(num_time_steps):
        # Reset the RHS
        b.x.array[:] = 0
        # Assemble the RHS => construct Au = b
        assemble_vector(b.x.petsc_vec, compiled_L)
        # Apply the Dirichlet boundary conditions
        apply_lifting(b.x.petsc_vec, [compiled_a], [[bc]])
        set_bc(b.x.petsc_vec, [bc])
        # Solve the linear system
        solver.solve(b.x.petsc_vec, uh.x.petsc_vec)
        uh.x.scatter_forward()
        set_bc(uh.x.petsc_vec, [bc])
        # Update the old values
        u_old.x.array[:] = uh.x.array.copy()
        # Store displacement (either as an array or a copy of uh)
        #displacement_history.append(uh.x.array)
        displacement_history.append(uh.copy())
    return displacement_history