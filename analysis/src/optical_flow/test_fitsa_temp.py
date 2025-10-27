"""Testing a temporal version of FISTA"""

from typing import List, Tuple, Union
import numpy as np
import scipy.sparse as sp
import numba
from scipy.ndimage import map_coordinates
from scipy import ndimage as ndi
from skimage.transform import pyramid_reduce # pylint: disable=no-name-in-module


def A(I):
    """
    Computes A such that A @ h = nabla_I ⋅ h

    Args:
        I (np.ndarray): The image whose gradient is computed (2D or 3D).
    
    Returns:
        sp.csr_matrix: Sparse matrix representing A.
    """
    grads = np.gradient(I) 
    grads_flat = [np.ravel(g, order='C') for g in grads]

    A_blocks = [sp.diags(g, offsets=0, format="csr") for g in grads_flat]

    A_full = sp.hstack(A_blocks, format="csr")

    return A_full

@numba.njit()
def nabla_block(S: Tuple[int, ...], axis=0, offset_i=0, offset_j=0) -> Tuple[List[int], List[int], List[int]]:
    """
    Constructs the sparse representation (row, col, value lists) of a discrete forward finite-difference 
    operator along a single spatial axis for a given multidimensional grid.

    The function creates one "block" of the full gradient operator — i.e., the partial derivative
    with respect to the chosen axis — as lists of indices and values that can later be assembled 
    into a sparse matrix.

    Args:
        S (Tuple[int, ...]): Shape of the multidimensional grid (e.g., `(ny, nx)` for a 2D image or `(nz, ny, nx)` for 3D).
        axis (int, optional): Axis along which to compute the discrete derivative.
            Defaults to 0.
        offset_i (int, optional): Row index offset for the resulting block in the global sparse matrix.
            Defaults to 0.
        offset_j (int, optional): Column index offset for the resulting block in the global sparse matrix.
            Defaults to 0.

    Returns:
        Tuple[List[int], List[int], List[int]]: Three lists `(rows, cols, values)` representing the COO-format entries of the 
            discrete difference operator:
              - `rows[i]`: row index of entry i  
              - `cols[i]`: column index of entry i  
              - `values[i]`: numerical value (+1 or -1)
    """
    num_pixels = 1
    step_size = 1
    temp_size = 1
    for ax, size in enumerate(S):
        num_pixels *= size
        if ax > axis:
            step_size *= size
        if ax < axis:
            temp_size *= size
 
    rows = []
    cols = []
    values = []
 
    for i in range(num_pixels):
        rows.append(i + offset_i)
        cols.append(i + offset_j)
        values.append(1)
 
        k = (i % (step_size * S[axis])) // step_size
 
        if k + 1 < S[axis]:
            rows.append(i + offset_i)
            cols.append(i + step_size + offset_j)
            values.append(-1)
 
    return rows, cols, values
 
 
def nabla(C, S: Tuple[int, ...]):
    """
    Constructs a full sparse gradient operator for a given multidimensional domain and 
    number of channels.

    The operator maps flattened input fields of shape `(C, *S)` to their discrete gradients 
    along each spatial axis, producing an array of shape `(C * len(S), *S)` when applied.

    Args:
        C (int): 
            Number of channels (e.g., components of a vector field such as u_x, u_y, u_z).
        S (Tuple[int, ...]): 
            Shape of the spatial domain (e.g., `(ny, nx)` for 2D, `(nz, ny, nx)` for 3D).

    Returns:
        sp.csr_array:
            Sparse matrix of shape `(C * len(S) * np.prod(S), C * np.prod(S))` in CSR format, 
            representing the discrete gradient operator ∇ applied to a multi-channel field.

    Notes:
        - The operator is assembled by stacking the per-axis finite-difference blocks generated 
          by `nabla_block`.
    """
    rows = []
    cols = []
    values = []
    num_pixels = np.prod(S, initial=1)
    for axis in range(len(S)):
        for channel in range(C):
            block_id = C * axis + channel
            rows_, cols_, values_ = nabla_block(
                S, axis=axis, offset_i=block_id * num_pixels, offset_j=channel * num_pixels
            )
            rows.extend(rows_)
            cols.extend(cols_)
            values.extend(values_)
 
    return sp.coo_array((values, (rows, cols)), shape=(len(S) * C * num_pixels, C * num_pixels)).tocsr()


def compute_Q(I: np.ndarray, alpha: float, beta:float, nabla_for_L: sp.spmatrix):
    """
    Computed the matrix Q defined as 2*(A^T*A + alpha*2*∇^T*∇ + beta*2*H^T*H)

    Args:
        I (np.ndarray): Initial image to compute A with
        alpha (float): Regularization parameter for the gradient
        beta (float): Regularization parameter for the hessian
        nabla_for_L (sp.spmatrix): Matrix representation of the gradient operator

    Returns:
         sp.csr_array:
            Sparse matrix reqpresenting Q
    """
    A_mat = A(I)
    nabla_mat = nabla_for_L
    H_mat = nabla_mat.T @ nabla_mat

    Q = A_mat.T @ A_mat + alpha**2 * (nabla_mat.T @ nabla_mat) + beta**2 * (H_mat.T @ H_mat)
    return 2*Q


def compute_lip(I: np.ndarray, alpha: float, beta:float, nabla_for_L: sp.spmatrix):
    ''' 
    Computes the best Lipschitz constant L of the gradient of the function 
    f(h) = norm_2(∇I2 ·(h - h0)+It)^2 + alpha * norm_2,1(∇h)^2 + beta * norm_2,1(Hh)^2
    L = sigma_max{2*(A^T*A + alpha*2*∇^T*∇ + beta*2*H^T*H)}
    
    Args:
        I (np.ndarray): The image whose gradient will be computed in A
        alpha (float): Regularization parameter controlling smoothness of the flow field.
        beta (float): Regularization parameter controlling smoothness of the gradient of the flow field.
        nabla_fot_L (np.ndarray):
        
    Returns:
        float: Theoretical Lipschitz constant of the gradient of f(h)
    '''
    Q = compute_Q(I, alpha, beta, nabla_for_L)
    return sp.linalg.norm(Q, ord=2)

def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warps a multi dimensionnal image using an optical flow

    Args:
        image (np.ndarray): Image to warp. Can optionnally have a channel dimension.
            Shape: (H, ...[, C])
        flow (np.ndarray): Optical flow. The first dimension is 2 for 2D images, 3 for 3D images, etc...
            Shape: (ndim, H, ...), dtype: float (32 or 64)

    Returns:
        np.ndarray: Warped image
            Shape: (H, ...[, C])
    """
    assert (
        flow.ndim - 1 == flow.shape[0]
    ), "First dimension should match the number of trailing dimensions"

    if image.ndim == flow.ndim:
        *dims, channels = image.shape
        points = np.indices(dims, flow.dtype) + flow

        return np.concatenate(
            [
                map_coordinates(image[..., channel], points, mode="nearest", order=1)[
                    ..., None
                ]
                for channel in range(channels)
            ],
            axis=-1,
        )

    # No channels
    assert image.ndim == flow.ndim - 1

    return map_coordinates(
        image, np.indices(image.shape, flow.dtype) + flow, mode="nearest", order=1
    )


def resize_flow(flow, shape):
    """
    Rescales the values of the vector field (u, v) to the desired shape.
    The values of the output vector field are scaled to the new resolution.
  
    Args:
        flow (np.ndarray): The displacement field
            Shape: # compléter
        shape (iterable): Couple of integers representing the output shape
        
    Returns:
        np.ndarray: The resized and rescaled motion field
    """

    scale = [n / o for n, o in zip(shape, flow.shape[1:])]
    scale_factor = np.array(scale, dtype=flow.dtype)

    for _ in shape:
        scale_factor = scale_factor[..., np.newaxis]

    rflow = scale_factor * ndi.zoom(
        flow, [1] + scale, order=1, mode="nearest", prefilter=False
    )

    return rflow


def get_pyramid(image, downscale=2.0, nlevel=10, min_size=16):
    """
    Construct image pyramid.
    
    Args:
        img (np.ndarray): The image to be preprocessed (Gray scale or RGB)
            Shape: # A compléter
        downscale (float): The pyramid downscale factor
            Default: 2
        nlevel (int): The maximum number of pyramid levels
            Default: 10
        min_size (int): The minimum soze for any dimension of the pyramid levels
            Default: 16
            
    Returns : 
        list[ndarray]: The coarse to fine images pyramid
    """

    pyramid = [image]
    size = min(image.shape)
    count = 1

    while (count < nlevel) and (size > downscale * min_size):
        J = pyramid_reduce(pyramid[-1], downscale)
        pyramid.append(J)
        size = min(J.shape)
        count += 1

    return pyramid[::-1]

 
def forward_diff(f: np.ndarray, axis: int) -> np.ndarray:
    ''' 
    Computes the forward difference of an array along a given axis.
    For element i: diff[i] = f[i+1] - f[i]
    The last element along the axis is set to zero.

    Args:
        f (np.ndarray): Input array
        axis (int): Axis along which to compute the difference

    Returns:
        np.ndarray: Array of the same shape as f containing forward differences
    '''
    if axis < 0 or axis >= f.ndim:
        raise ValueError(f"axis must be between 0 and {f.ndim}")

    diff = np.zeros_like(f)

    slc_start = [slice(None)] * f.ndim
    slc_end = [slice(None)] * f.ndim
    
    # Select slices for forward difference
    slc_start[axis] = slice(0, -1)
    slc_end[axis] = slice(1, None)

    # Forward difference : f[i+1] - f[i]
    diff[tuple(slc_start)] = f[tuple(slc_end)] - f[tuple(slc_start)]
    
    # Last element remains zero

    return diff


def backward_diff(f: np.ndarray, axis: int) -> np.ndarray:
    ''' 
    Computes the backward difference of an array along a given axis.
    For element i: diff[i] = f[i] - f[i-1]
    Special handling at the boundaries:
        diff[0] = f[0]
        diff[-1] = -f[-2]

    Args:
        f (np.ndarray): Input array
        axis (int): Axis along which to compute the difference

    Returns:
        np.ndarray: Array of the same shape as f containing backward differences
    '''
    if axis < 0 or axis >= f.ndim:
        raise ValueError(f"axis must be between 0 and {f.ndim - 1}")

    diff = np.zeros_like(f)

    slc_prev = [slice(None)] * f.ndim
    slc_curr = [slice(None)] * f.ndim

    slc_prev[axis] = slice(0, -1)
    slc_curr[axis] = slice(1, None)

    # Main backward difference: f[i] - f[i-1]
    diff[tuple(slc_curr)] = f[tuple(slc_curr)] - f[tuple(slc_prev)]

    # diff[0] = f[0]
    start_idx = [slice(None)] * f.ndim
    start_idx[axis] = 0
    diff[tuple(start_idx)] = f[tuple(start_idx)]

    # diff[-1] = -f[-2]
    end_idx = [slice(None)] * f.ndim
    end_idx[axis] = -1
    prev_idx = [slice(None)] * f.ndim
    prev_idx[axis] = -2
    diff[tuple(end_idx)] = -f[tuple(prev_idx)]

    return diff


def nabla_h(h: np.ndarray, gamma: float = None) -> np.ndarray:
    ''' 
    Computes the discrete gradient (forward differences) of a d-dimensional flow field.
    
    Args:
        h (np.ndarray): Input array
            Shape: (d, T, *S)
        gamma (float, optional): Weight for the temporal gradient

    Returns:
        np.ndarray: Array containing gradients of each component along each axis
    '''

    d = h.shape[0]  # displacement components
    
    ndim = h.ndim - 1  # number of axes excluding displacement index
    
    grads = np.zeros((d, ndim) + h.shape[1:], dtype=h.dtype)

    for i in range(d):
        for axis in range(ndim):
            # Forward differences along each axis for each component, including time
            grads[i, axis] = forward_diff(h[i], axis=axis)
            if axis==0 and gamma is not None:
                grads[i, axis] *= gamma

    return grads


def nabla_star_h(grads: np.ndarray) -> np.ndarray:
    ''' 
    Computes the discrete divergence (backward differences) of a d-dimensional gradient field.
    This is the adjoint of nabla_h.

    Args:
        grads (np.ndarray): Gradient array 
            Shape: (d, d, T, *S)

    Returns:
        np.ndarray: Array containing the divergence of each component
            Shape: (d, T, *S) 
    '''
    
    d = grads.shape[0]
    ndim = grads.shape[1]
    div = np.zeros((d,) + grads.shape[2:], dtype=grads.dtype)

    for i in range(d):
        for axis in range(ndim):
            # Backward difference along each axis, summed over components, inclusing time
            div[i] -= backward_diff(grads[i, axis], axis=axis)
    
    return div


def hessian_(h: np.ndarray, nablah=np.ndarray, gamma: float=None) -> np.ndarray:
    ''' 
    Computes the discrete Hessian of a d-dimensional flow field.
    Uses forward differences twice: first to get gradients, then to get second derivatives.

    Args:
        h (np.ndarray): Input array 
            Shape: (d, T, *S)
        gamma (float): Weight for the temporal gradient

    Returns:
        np.ndarray: Array containing second derivatives along each axis
            Shape: (d, d, d, T, *S)
    '''
    d = h.shape[0]
    ndim = h.ndim - 1

    hessian = np.zeros((d,ndim,ndim) + h.shape[1:], dtype=h.dtype)
    
    for i in range(d):
        for axisder in range(ndim):
            for axishess in range(ndim):
                # Forward difference of the first derivative gives second derivative
                hessian[i, axisder, axishess] = forward_diff(nablah[i, axisder], axis=axishess)
                if axishess == 0 and gamma is not None: 
                    hessian[i, axisder, axishess] *= gamma
                    
    return hessian


def hessian_star_h(hess: np.ndarray) -> np.ndarray:
    '''
    Computes the adjoint operator of the discrete Hessian

    Args:
        hess (np.ndarray): Discrete Hessian tensor
            Shape: (d, d, d, T, *S)

    Returns:
        np.ndarray: Array corresponding to the adjoint
                    action of the Hessian on the vector field.
            Shape: (d, T, *S)
    '''
    d = hess.shape[0]
    ndim = hess.shape[1]
    spatial_shape = hess.shape[3:]
    
    adjoint = np.zeros((d,) + spatial_shape, dtype=hess.dtype)
    
    for i in range(d):
        for axis1 in range(ndim):
            for axis2 in range(ndim): 
                # Apply backward difference twice:
                # - once along the direction of last derivative
                # - once along direction of first derivative
                adjoint[i] += -backward_diff(-backward_diff(hess[i, axis1, axis2], axis=axis2), axis=axis1)
    
    return adjoint


def nabla_f(
    h:np.ndarray, 
    h0: np.ndarray, 
    It: np.ndarray, 
    nablaI: np.ndarray, 
    alpha: float, 
    beta: float,
    gamma: float = None) -> np.ndarray:
    ''' 
    Computes the gradient of the energy functional f(h) at the current estimate h.
    The functional includes:
        - Data term: ||It + nablaI · (h - h0)||^2
        - Gradient regularization: alpha^2 * ||∇h||^2
        - Hessian regularization: beta^2 * ||∇²h||^2  (if beta ≠ 0)

    Args:
        h (np.ndarray): Current displacement field 
            Shape: (d, *S)
        h0 (np.ndarray): Reference displacement field (outer iteration)
            Shape: (d, *S)
        It (np.ndarray): Temporal difference image (ref - warped(moving))
            Shape: (*S)
        nablaI (np.ndarray): Spatial gradients of the warped moving image
            Shape (d, *S)
        alpha (float): Regularization weight for gradient smoothness
        beta (float): Regularization weight for Hessian smoothness

    Returns:
        np.ndarray: Gradient of f(h)
            Shape: (d, *S)
    '''
    # Gradient of data attached term
    proj = (nablaI * (h - h0)).sum(axis=0)
    grad_data = 2 * nablaI * (It + proj)

    # Gradient of the gradient regularization term
    nablah = nabla_h(h, gamma)
    grad_reg = 2 * (alpha**2) * nabla_star_h(nablah)
    
    # Gradient of the hessian regularization term
    grad_reg_hess = 0
    
    if beta != 0:
        hessianh = hessian_(h, nablah, gamma)
        grad_reg_hess = 2 * (beta**2) * hessian_star_h(hessianh)
    
    return grad_data + grad_reg + grad_reg_hess


def p_L(
    h:np.ndarray, 
    h0: np.ndarray, 
    lipschitz_constant: float,
    It: np.ndarray, 
    nablaI: np.ndarray,
    alpha: float, 
    beta: float,
    gamma: float = None) -> np.ndarray:
    ''' 
    Single proximal gradient for FISTA step.

    Args:
        h (np.ndarray): Current displacement field
            Shape: (d, *S)
        h0 (np.ndarray): Reference displacement field (outer iteration)
            Shape: (d, *S)
        lipschitz_constant (float): Lipschitz constant of ∇f
        It (np.ndarray): Temporal difference image
            Shape: (*S)
        alpha (float): Regularization weight for gradient smoothness
        beta (float): Regularization weight for Hessian smoothness
        nablaI (np.ndarray): Spatial gradients of warped moving image
            Shape: (d, *S)

    Returns:
        np.ndarray: Updated displacement field after one proximal step
            Shape: (d, *S)
    '''
    nabla_f_ = nabla_f(h, h0, It, nablaI, alpha, beta, gamma)
    return h - (1/lipschitz_constant) * nabla_f_


def _fista(    
    ref: np.ndarray,
    moving: np.ndarray,
    h0: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0,
    num_iter: int = 100,
    num_warp: int = 2,
    eps: float = 1e-5) -> np.ndarray:
    ''' 
    Computes the optical flow between two images (reference and moving) using
    FISTA algorithm with iterative warping

    Args:
        ref (np.ndarray): Reference image
            Shape: (*S)
        moving (np.ndarray): Moving image to be aligned
            Shape: (*S)
        h0 (np.ndarray): Initial displacement field
            Shape: (d, *S)
        alpha (float): Weight for gradient regularization
        beta (float): Weight for Hessian regularization
        lipschitz_constant (float): Lipschitz constant (step size denominator)
        num_iter (int): Number of FISTA iterations per warp step
        num_warp (int): Number of warping updates
        eps (float): Stopping criterion threshold

    Returns:
        np.ndarray: Optimized displacement field h
            Shape: (d, *S)

    References:
        Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. 
        SIAM Journal on Imaging Sciences, 2(1), 183–202
    '''
    
    ref = ref.astype(np.float32)
    moving = moving.astype(np.float32)
    h0 = h0.astype(np.float32)
    h = h0.copy()
    
    # FISTA auxiliary variable
    y = h.copy()
    #import matplotlib.pyplot as plt
    
    # FISTA momentum term
    t = 1
    t_new = t
    
    H, W = ref.shape
    nabla_for_l = nabla(2, (H, W))
    
    for _ in range(num_warp):
        # Warp the moving image according to the current displacement estimate h0
        im2_warped = warp(moving, h0)

        # Compute spatial gradients of the warped image
        nablaI = np.array(np.gradient(im2_warped))
        # Compute temporal intensity difference between reference and warped image
        It = im2_warped - ref
        
        lipschitz_constant = compute_lip(im2_warped, alpha, beta, nabla_for_l)

        for _ in range(num_iter):
            # Proximal gradient update 
            h_new = p_L(y, h0, lipschitz_constant, It, nablaI, alpha, beta)
            
            # Update momentum parameter
            t_new = (1/2) * (1 + np.sqrt(1 + 4 * (t**2)))
            
            # Extrapolation step
            factor = (t - 1) / t_new 
            y = h_new + factor * (h_new - h)
            
            # Convergence check based on mean squared change in h
            if np.linalg.norm(y - h)**2 <= eps**2: 
                break
            
            # Update h and t for new inner iteration
            h = h_new.copy()
            t = t_new.copy()
        
        # Update initial displacement field for next outer iteration
        h0 = h.copy()
    
    return h0


def fista(    
    reference: np.ndarray,
    moving: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0,
    num_iter: int = 100,
    num_warp: int = 2,
    num_pyramid: int = 10,
    pyramid_downscale: Union[float, np.ndarray] = 2.0,
    pyramid_min_size: Union[int, np.ndarray] = 16,
    eps: float = 1e-5) -> np.ndarray:
    ''' 
    Computes optical flow between two images (reference and moving) using
    FISTA algorithm with a multi-scale image pyramid for efficient
    large-displacement handling.

    The algorithm works by:
        1. Building pyramids of the reference and moving images.
        2. Estimating the displacement field (flow) starting at the coarsest level.
        3. Progressively refining the flow at finer levels using FISTA with warping.

    Args:
        reference (np.ndarray): Reference image
            Shape: (*S)
        moving (np.ndarray): Moving image to be aligned to the reference
            Shape: (d, *S)
        alpha (float, optional): Regularization weight for gradient smoothness
            Default: 0.01
        beta (float, optional): Regularization weight for Hessian smoothness
            Default: 0
        lipschitz_constant (float, optional): Lipschitz constant (step size denominator)
            Default: 1000
        num_iter (int, optional): Number of FISTA iterations per warp step (inner iterations)
            Default: 100
        num_warp (int, optional): Number of warp updates per pyramid level (outer iterations)
            Default: 2
        num_pyramid (int, optional): Maximum number of pyramid levels
            Default: 10
        pyramid_downscale (Union[float, np.ndarray], optional): Scaling factor or array for downsampling
                            at each pyramid level.
            Default: 2.0
        pyramid_min_size (Union[int, np.ndarray], optional): Minimum size of images in the pyramid
            Default: 16
        eps (float, optional): Stopping criterion threshold
            Default: 1e-5

    Returns:
        np.ndarray: Estimated displacement field h, same shape as input images
    
    References:
        Meinhardt-Llopis, E., & Sánchez, J. (2013). Horn-schunck optical flow with a multi-scale strategy.
        Image Processing on line.
    '''
    
    # Input images must be of the same shape
    assert moving.shape == reference.shape
    
    # Dimension of the input images (e.g., 2D or 3D)
    d = reference.ndim

    # Build the image pyramids for both reference and moving images
    pyramid_ = list(
        zip(
            get_pyramid(reference, pyramid_downscale, num_pyramid, pyramid_min_size),
            get_pyramid(moving, pyramid_downscale, num_pyramid, pyramid_min_size),
        )
    )

    # Initialize displacement field (optical flow) at the coarsest pyramid level
    h0 = np.zeros(
        (d,) + pyramid_[0][0].shape
    )  # Shape matching the coarsest reference level

    # Compute optical flow at the coarsest pyramid level
    h = _fista(
        pyramid_[0][0],
        pyramid_[0][1],
        h0,
        alpha,
        beta,
        num_iter=num_iter,
        num_warp=num_warp,
        eps=eps
    )

    # Progressively refine the optical flow up the pyramid levels
    for pyr_ref, pyr_mov in pyramid_[1:]:
        # Resize the flow field to the current pyramid level's dimensions
        h = resize_flow(h, pyr_ref.shape)

        # Update the flow field using the HS algorithm at the current pyramid level
        h = _fista(
            pyr_ref,
            pyr_mov,
            h,
            alpha,
            beta,
            num_iter=num_iter,
            num_warp=num_warp,
            eps=eps
        )

    return h




def _fista_global(    
    image: np.ndarray,
    h0: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0,
    gamma: float = 0,
    num_iter: int = 100,
    num_warp: int = 2,
    eps: float = 1e-5) -> np.ndarray:
    ''' 
    Computes the optical flow between two images (reference and moving) using
    FISTA algorithm with iterative warping

    Args:
        image (np.ndarray): Images
            Shape: (T, *S)
        h0 (np.ndarray): Initial displacement field
            Shape: (d, T, *S)
        alpha (float): Weight for gradient regularization
        beta (float): Weight for Hessian regularization
        gamma (float): Weight for temporal regularization
        L (float): Lipschitz constant (step size denominator)
        num_iter (int): Number of FISTA iterations per warp step
        num_warp (int): Number of warping updates
        eps (float): Stopping criterion threshold

    Returns:
        np.ndarray: Optimized displacement field h
            Shape: (d, T, *S)

    References:
        Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. 
        SIAM Journal on Imaging Sciences, 2(1), 183–202
    '''
    
    h = h0.copy()
    h_new = np.zeros_like(h0)
    h0 = h0.astype(np.float32)
    h = h0.copy()
    
    # FISTA auxiliary variable
    y = h.copy()
    
    # FISTA momentum term
    t = 1
    t_new = t
    
    nbflows = image.shape[0] - 1
    
    for _ in range(num_warp):
        # Warp the moving image according to the current displacement estimate h0
        im2_warped = np.stack([warp(image[t], h0[:,t-1]) for t in range(1, image.shape[0])])
        
        # Compute the spatial gradient of the warped images
        nablaI = np.stack(np.gradient(im2_warped, axis=tuple(range(1, im2_warped.ndim))))

        # Compute temporal intensity difference between reference and warped image
        It = image[:nbflows] - im2_warped 
        
        for nb in range(num_iter):
            # Proximal gradient update 
            h_new = p_L(y, h0, L, It, nablaI, alpha, beta, gamma)
            
            # Update momentum parameter
            t_new = (1/2) * (1 + np.sqrt(1 + 4 * (t**2)))
            
            # Extrapolation step
            factor = (t - 1) / t_new 
            y = h_new + factor * (h_new - h)
            
            # Convergence check based on mean squared change in h
            if np.linalg.norm(h_new - h)**2 <= eps**2: 
                break
            
            # Some optional printing to see the evolution
            #if nb == (num_iter-1):
                #print(np.linalg.norm(h_new - h)**2)
            
            # Update h and t for new inner iteration
            h = h_new.copy()
            t = t_new.copy()
        
        # Update initial displacement field for next outer iteration
        h0 = h.copy()
    
    return h0    



def fista_global(    
    image: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0,
    gamma: float = 0,
    num_iter: int = 100,
    num_warp: int = 2,
    num_pyramid: int = 10,
    pyramid_downscale: Union[float, np.ndarray] = 2.0,
    pyramid_min_size: Union[int, np.ndarray] = 16,
    eps: float = 1e-5) -> np.ndarray:
    ''' 
    Computes optical flow between two images (reference and moving) using
    FISTA algorithm with a multi-scale image pyramid for efficient
    large-displacement handling.

    The algorithm works by:
        1. Building pyramids of the reference and moving images.
        2. Estimating the displacement field (flow) starting at the coarsest level.
        3. Progressively refining the flow at finer levels using FISTA with warping.

    Args:
        image (np.ndarray): images
            Shape: (T, *S)
        alpha (float, optional): Regularization weight for gradient smoothness
            Default: 0.01
        beta (float, optional): Regularization weight for Hessian smoothness
            Default: 0
        gamma (float, optioncal): Regularization weight for temporal smoothness
            Default: 0
        L (float, optional): Lipschitz constant (step size denominator)
            Default: 1000
        num_iter (int, optional): Number of FISTA iterations per warp step (inner iterations)
            Default: 100
        num_warp (int, optional): Number of warp updates per pyramid level (outer iterations)
            Default: 2
        num_pyramid (int, optional): Maximum number of pyramid levels
            Default: 10
        pyramid_downscale (Union[float, np.ndarray], optional): Scaling factor or array for downsampling
                            at each pyramid level.
            Default: 2.0
        pyramid_min_size (Union[int, np.ndarray], optional): Minimum size of images in the pyramid
            Default: 16
        eps (float, optional): Stopping criterion threshold
            Default: 1e-5

    Returns:
        np.ndarray: Estimated displacement field h, same shape as input images
    
    References:
        Meinhardt-Llopis, E., & Sánchez, J. (2013). Horn-schunck optical flow with a multi-scale strategy.
        Image Processing on line.
    '''
    
    if image.shape[0] < 3 : 
        print("Not enough frames for temporal regularisation")
        dimension = len(image[0].shape)
        h = np.zeros((dimension, image.shape[0]) + image[0].shape)

        for frame in range(image.shape[0]-1):
            h[:,frame] = fista(image[frame], image[frame+1], alpha=alpha, beta=beta, num_iter=num_iter, num_warp=num_warp, num_pyramid=num_pyramid, pyramid_downscale=pyramid_downscale, pyramid_min_size=pyramid_min_size, eps=eps)
        return h
    
    if gamma == 0 or gamma is None :
        print("No temporal regularization")
        dimension = len(image[0].shape)
        h = np.zeros((dimension, image.shape[0]-1) + image[0].shape)

        for frame in range(image.shape[0]-1):
            h[:,frame] = fista(image[frame], image[frame+1], alpha=alpha, beta=beta, num_iter=num_iter, num_warp=num_warp, num_pyramid=num_pyramid, pyramid_downscale=pyramid_downscale, pyramid_min_size=pyramid_min_size, eps=eps)
    
    else : 
        # Build the image pyramids for all the images
        if image.ndim == 3:
            pyramid_list = [get_pyramid(image[t], pyramid_downscale, num_pyramid, pyramid_min_size) for t in range(image.shape[0])]
            pyrlist = [np.stack(levels_at_all_t, axis=0) for levels_at_all_t in zip(*pyramid_list)]

        else:
            pyramid_list = [
            [get_pyramid(image[t, z], pyramid_downscale, num_pyramid, pyramid_min_size)
            for z in range(image.shape[1])]
            for t in range(image.shape[0])
            ]

            # Reformatage : pyrlist[lvl] aura la forme (T, Z, H_l, W_l)
            pyrlist = []
            num_levels = len(pyramid_list[0][0])
            for l in range(num_levels):
                level_images = np.stack([
                    np.stack([pyramid_list[t][z][l] for z in range(image.shape[1])])
                    for t in range(image.shape[0])
                ])
            pyrlist.append(level_images)
            
        coarse_shape = pyrlist[0][0].shape 
        dim = image.ndim

        # Initialize displacement field (optical flow) at the coarsest pyramid level
        h0 = np.zeros(((dim-1, image.shape[0] - 1) + coarse_shape))  # Shape matching the coarsest reference level

        # Compute optical flow at the coarsest pyramid level
        h = _fista_global(
            pyrlist[0],
            h0,
            alpha,
            beta,
            gamma,
            num_iter=num_iter,
            num_warp=num_warp,
            eps=eps
        )

        # Loop from coarsest to finest
        for lvl, pyr_image in enumerate(pyrlist[1:]):
            print('new pyramid of shape : ', pyr_image.shape)
            # Upscale flow to current level if not first level
            if lvl+1 > 0:
                h = np.swapaxes(np.stack([resize_flow(h[:, t], pyr_image[t].shape) for t in range(h.shape[1])]), 0, 1)

            # Update the flow field using the HS algorithm at the current pyramid level
            h = _fista_global(
                pyr_image,
                h,
                alpha,
                beta,
                gamma,
                num_iter=num_iter,
                num_warp=num_warp,
                eps=eps
            )

    return h