from typing import Tuple
import numpy as np 
from dolfinx import default_scalar_type
import dolfinx.fem as fem
import dolfinx.mesh as mesh 
import ufl
from ufl import TrialFunction, TestFunction, inner, grad, adjoint, derivative, action, Measure
from mechanics.src.utils import compute_lame
from data_generation.src.fem.solver import create_dirichlet, strain, stress_elas, stress_visc
from scipy.optimize import minimize
from data_generation.src.imaging.generator import dirichlet
from petsc4py import PETSc

def array_to_function(
    V: fem.functionspace, 
    h: np.ndarray, 
    msh: mesh.Mesh, 
    x_range: Tuple[float, float], 
    y_range: Tuple[float, float]):
    
    Ny, Nx = h.shape[2:]
    xmin, xmax = x_range
    ymin, ymax = y_range
    
    dof_coords = V.tabulate_dof_coordinates().reshape(-1, msh.geometry.dim)
    num_nodes = dof_coords.shape[0]
    bs = V.dofmap.index_map_bs 

    values2 = np.zeros((num_nodes, bs), dtype=np.float64)

    for node in range(num_nodes):
        x, y, _ = dof_coords[node]

        ix = int(np.clip((x - xmin) / (xmax - xmin) * (Nx - 1), 0, Nx - 1))
        iy = int(np.clip((y - ymin) / (ymax - ymin) * (Ny - 1), 0, Ny - 1))

        for comp in range(bs):
            if comp < 2:
                values2[node, comp] = h[comp-1, 0, iy, ix]
            else:
                values2[node, comp] = 0.0

    values = values2.ravel()

    f = fem.Function(V)
    f.x.array[:] = values
    f.x.scatter_forward()
    return f

def get_E(
    ym_init, 
    h_obs_array, 
    x_range, 
    y_range, 
    msh, 
    traction_zone, 
    alpha_value, 
    zone_radius, 
    zone_center, 
    x_c, 
    y_c, 
    physical_length, 
    bounds, 
    iter_for_min, 
    ftol, 
    gtol
):
    V = fem.functionspace(msh, ('Lagrange', 2, (msh.geometry.dim,)))
    Q = fem.functionspace(msh, ("CG", 1))
    
    u = fem.Function(V, name="u")
    lmbda = fem.Function(V, name="lambda")
    E = fem.Function(Q, name="E")
    
    E.x.array[:] = ym_init 
    E.x.scatter_forward()
    
    nu = 0.3
    traction_constant = fem.Constant(msh, default_scalar_type((0, traction_zone, 0)))
    h_obs = array_to_function(V, h_obs_array, msh, x_range, y_range)
    
    ds = Measure("ds", domain=msh)
    dx = Measure("dx", domain=msh)
    
    alpha = fem.Constant(msh, alpha_value)
    bc = create_dirichlet(msh, dirichlet, V, zone_radius, zone_center, x_c, y_c, physical_length)
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    mu, lmbda_lame = compute_lame(E, nu)

    J = inner(u - h_obs, u - h_obs) * dx + alpha * ufl.sqrt(inner(grad(E), grad(E)) + 1e-6) * dx #+ alpha*inner(grad(E), grad(E))*dx #
    R = (ufl.inner(stress_elas(u, lmbda_lame, mu), strain(v_test)) * dx - ufl.inner(traction_constant, v_test) * ds)
    
    a = ufl.inner(stress_elas(u_trial, lmbda_lame, mu), strain(v_test)) * dx
    L = ufl.inner(traction_constant, v_test) * ds

    forward_problem = fem.petsc.LinearProblem(a, L,
                                            bcs=[bc],
                                            petsc_options_prefix="fwd",
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                            u=u)

    du = ufl.TrialFunction(V)

    adjoint_lhs = ufl.adjoint(ufl.derivative(R, u, du))
    adjoint_rhs = ufl.derivative(J, u, v_test)

    adjoint_problem = fem.petsc.LinearProblem(adjoint_lhs, adjoint_rhs,
                                                bcs=[bc],
                                                petsc_options_prefix="adj",
                                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                                u=lmbda)
    
    def J_and_grad_for_E(E_array):
        E.x.array[:] = E_array
        E.x.scatter_forward()
        
        forward_problem.solve()
        J_val = fem.assemble_scalar(fem.form(J))
        
        adjoint_problem.solve()

        dJdE = fem.form(
            ufl.derivative(J, E)
            - ufl.action(ufl.adjoint(ufl.derivative(R, E)), lmbda)
        )

        grad_vec = fem.petsc.assemble_vector(dJdE)
        grad_vec.assemble()
        
        return J_val, grad_vec

    def callback(intermediate_result):
        fval = intermediate_result.fun
        print(f"J: {fval}")
    
    result = minimize(
        J_and_grad_for_E,
        x0=E.x.array.copy(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": iter_for_min, "ftol": ftol, "gtol": gtol},
        bounds=bounds * E.x.array.size, 
        callback=callback, 
        tol = 1e-9
    )
    
    return result


def get_eta(
    eta_init_, 
    ym_found_, 
    nb_steps, 
    h_obs_array, 
    x_range, 
    y_range, 
    msh, 
    traction_zone, 
    alpha_value, 
    zone_radius, 
    zone_center, 
    x_c, 
    y_c, 
    physical_length, 
    t_end, 
    num_time_steps, 
    bounds, 
    iter_for_min, 
    ftol, 
    gtol):
    
    V = fem.functionspace(msh, ('Lagrange', 2, (msh.geometry.dim,)))
    Q = fem.functionspace(msh, ("CG", 1))
    ym = fem.Function(Q)
    ym.x.array[:] = ym_found_.x.copy()
    
    eta = fem.Function(Q)
    eta.x.array[:] = eta_init_
    
    nu = 0.3
    delta_t = float(t_end/num_time_steps)
    
    traction = fem.Constant(msh, default_scalar_type((0, traction_zone, 0)))
    alpha = fem.Constant(msh, alpha_value)
    bc = create_dirichlet(msh, dirichlet, V, zone_radius, zone_center, x_c, y_c, physical_length)
        
    h_obs = [
        array_to_function(
            V, h_obs_array[:, i:i+1],
            msh, x_range, y_range
        )
        for i in range(h_obs_array.shape[1] - 1)
    ]
    
    ds = Measure("ds", domain=msh)
    dx = Measure("dx", domain=msh)
    

    def J_and_grad_for_eta(eta_array):
        eta = fem.Function(Q)
        eta.x.array[:] = eta_array
        eta.x.scatter_forward()
        
        dLdeta = None
        Reg =  alpha * ufl.sqrt(inner(grad(eta), grad(eta)) + 1e-6) * dx
        J = None
        u_old = fem.Function(V)
        
        for t in range(nb_steps):
            u_trial, v_test, u_t = TrialFunction(V), TestFunction(V), fem.Function(V)
            
            mu_v, lambda_v = eta / 2.0, -eta / 3.0
            mu_, lambda_ = compute_lame(ym, nu)
            a = inner(stress_elas(u_trial, lambda_, mu_) + stress_visc(u_trial, lambda_v, mu_v, delta_t), strain(v_test)) * dx
            L = inner(traction, v_test) * ds + inner(stress_visc(u_old, lambda_v, mu_v, delta_t), strain(v_test)) * dx
            
            R_t = inner(stress_elas(u_t, lambda_, mu_) + stress_visc(u_t, lambda_v, mu_v, delta_t), strain(v_test))*dx - (inner(traction, v_test) * ds + inner(stress_visc(u_old, lambda_v, mu_v, delta_t), strain(v_test)) * dx)
           
            problem_fwd = fem.petsc.LinearProblem(a, L,
                                                    bcs=[bc],
                                                    petsc_options_prefix="fwd",
                                                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                                    u=u_t)
            problem_fwd.solve()
            
            J_t = inner(u_t - h_obs[t], u_t - h_obs[t]) * dx
            
            lhs = fem.form(adjoint(derivative(R_t,u_t)))
            rhs = derivative(J_t, u_t)
            
            lmbda_t = fem.Function(V)
            problem_adj = fem.petsc.LinearProblem(lhs, rhs,
                                                    bcs=[bc],
                                                    petsc_options_prefix="adj",
                                                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                                    u=lmbda_t)
            problem_adj.solve()
            
            dLdeta_t = derivative(J_t, eta) + action(adjoint(derivative(R_t, eta)), lmbda_t)
            
            if J is None:
                dLdeta = dLdeta_t
                J = J_t
            else:
                dLdeta += dLdeta_t
                J += J_t
            
            u_old.x.array[:] = u_t.x.array
            u_old.x.scatter_forward()
            
        dLdeta += derivative(Reg, eta)
        
        grad_vec = fem.petsc.assemble_vector(fem.form(dLdeta))
        grad_vec.assemble()
        J_val = fem.assemble_scalar(fem.form(J + alpha * ufl.sqrt(inner(grad(eta), grad(eta)) + 1e-6) * dx))
        
        return float(J_val), grad_vec.array

    def callback(intermediate_result):
        fval = intermediate_result.fun
        print(f"J: {fval}")
    
    result = minimize(
        J_and_grad_for_eta,
        x0=eta.x.array.copy(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": iter_for_min, "ftol": ftol, "gtol": gtol},
        bounds=bounds * eta.x.array.size, 
        callback=callback, 
        tol = 1e-4
    )
    
    return result