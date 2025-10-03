############################################################
############################################################
############################################################


# def compute_projection_tensor_from_voxel_array(f):
#     ### Notes
#     ### - conventions are Cartesian-RHG, how to match with numpy-LHG?
    
#     nz, ny, nx = f.shape
    
#     ### EV: add option to change these if desired
#     n_points = nz  # number of Fourier grid points per 1D slice, scales with volume size
#     n_phi = nz  # number of projection angles in phi
#     n_theta = nz  # number of projection angles in theta
    
#     ### start with a central slice along the z-axis
#     ### rotate by theta (around y-axis) to form xz-plane of 1D slices
    
#     ### EV: TODO: define point spacing based on odd or even grid size (DFT conventions)
#     z = np.linspace(-(1 - (1/n_points)), 1 - (1/n_points), n_points, endpoint=True)  # initial Fourier points along z-axis (x3)
#     # z = np.linspace(-1, 1, n_points, endpoint=False)  # initial Fourier points along z-axis (x3)
#     # z = np.linspace(1 - (1/n_points), -(1 - (1/n_points)), n_points, endpoint=True)
    
#     theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
#     x_theta = z[:, np.newaxis] * np.sin(theta)[np.newaxis, :]
#     y_theta = np.zeros((n_points, n_theta))
#     z_theta = z[:, np.newaxis] * np.cos(theta)[np.newaxis, :]
    
#     ### now rotate each slice by phi (around the z-axis)
    
#     phi = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
#     x_theta_phi = x_theta[np.newaxis, ...] * np.cos(phi[:, np.newaxis, np.newaxis])
#     y_theta_phi = x_theta[np.newaxis, ...] * np.sin(phi[:, np.newaxis, np.newaxis])
#     z_theta_phi = np.ones((n_phi, n_points, n_theta)) * z_theta
    
#     ### format for fiNUFFT
    
#     x_theta_phi = np.pi * x_theta_phi.flatten()
#     y_theta_phi = np.pi * y_theta_phi.flatten()
#     z_theta_phi = np.pi * z_theta_phi.flatten()

#     nufft_type=2
#     eps=1e-8
#     n_transforms=1
    
#     plan = finufft.Plan(nufft_type, (nx, ny, nz), n_transforms, eps)
    
#     ### EV: not sure what correct ordering is to match conventions
#     plan.setpts(x_theta_phi, y_theta_phi, z_theta_phi)
    
#     f_hat_slices = plan.execute(f.astype(np.complex128))
#     f_hat_slices = f_hat_slices.reshape((n_phi, n_points, n_theta))
    
#     f_projection_tensor = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat_slices, axes=1), axis=1), axes=1).real

#     return f_projection_tensor


# def pdf_tensor_to_cdf_tensor(F):
   
#     ### EV: will need to add a normalization step for ramp filtered slices

#     return np.cumsum(F, axis=1)


# def cdf_tensor_to_icdf_tensor(C):
    
#     U = np.zeros(C.shape)
#     n_phi, nx, n_theta = C.shape
     
#     ### EV: add option later to change number of points and spacing if desired
#     x = np.linspace(0, 1, nx, endpoint=True)
    
#     ### EV: NEED TO VECTORIZE THIS
#     for phi in range(n_phi):
#         for theta in range(n_theta):
#             U[phi, :, theta] = np.interp(x, C[phi, :, theta], x)
    
#     ### EV: ENFORCE BOUNDS (not sure why this helps)
#     U[:, 0, :] = 0
#     U[:, -1, :] = 1

#     return U


# def compute_inverse_cdfs(P, x, u_grid):
#     """
#     Vectorized computation of inverse CDFs for a matrix P.
#     Each column of P is a PDF defined on grid x.
   
#     Parameters:
#       P      : 2D numpy array of shape (len(x), M), with each column a PDF.
#       x      : 1D numpy array of grid points.
#       u_grid : 1D numpy array of values in [0,1] for evaluating the inverse CDF.
     
#     Returns:
#       inv_cdfs: 2D array of shape (len(u_grid), M) where each column is the inverse CDF.
#     """
    
#     dx = x[1] - x[0]
#     # Normalize each column (if not already normalized)
#     P_norm = P / (np.sum(P, axis=0) * dx)
#     # Compute the CDFs along axis=0 (each column)
#     CDF = np.cumsum(P_norm, axis=0) * dx
#     # Normalize each column's CDF so that the last entry is 1
#     CDF = CDF / CDF[-1, :]
   
#     ny, nx = CDF.shape
#     cp = u_grid.size
    
#     ICDF = np.zeros((cp, nx))
    
#     for idx in range(nx):
#         # We set bounds_error=False and force fill_value=(x[0], x[-1]) so that u=0,1 map to x[0], x[-1].
#         interpolator = interp1d(CDF[:, idx], x, axis=0, bounds_error=False, fill_value=(x[0], x[-1]), assume_sorted=True)
#         ICDF[:, idx] = interpolator(u_grid)   
    
#     return ICDF


############################################################
############################################################
############################################################


# def clenshaw_curtis(n):
#     """
#     Compute Clenshaw-Curtis nodes and weights for integration on [0,1].
   
#     This function first computes Chebyshev nodes on [-1,1] using
#        x_i = cos(pi * i/(n-1)), i=0,..., n-1,
#     then transforms them to [0,1] via
#        u = (1 - x)/2.
#     The weights are computed using a standard algorithm.
   
#     Parameters:
#       n : int, number of nodes (n >= 2)
     
#     Returns:
#       u : 1D numpy array of nodes in [0,1] (nonuniform, clustering near 0 and 1)
#       w : 1D numpy array of quadrature weights (summing to 1)
#     """
    
#     if n < 2:
#         raise ValueError("n must be at least 2")
#     k = np.arange(n)
#     theta = np.pi * k / (n - 1)
#     x_cc = np.cos(theta)       # nodes on [-1,1]
#     u = (1 - x_cc) / 2         # transform to [0,1]
   
#     w = np.zeros(n)
#     if (n - 1) % 2 == 0:
#         m = (n - 1) // 2
#         for i in range(n):
#             s = 0.0
#             for j in range(1, m):
#                 s += 2 * np.cos(2 * j * theta[i]) / (4 * j**2 - 1)
#             s += np.cos(2 * m * theta[i]) / (4 * m**2 - 1)
#             w[i] = 2 / (n - 1) * (1 - s)
#     else:
#         m = (n - 1) // 2
#         for i in range(n):
#             s = 0.0
#             for j in range(1, m + 1):
#                 s += 2 * np.cos(2 * j * theta[i]) / (4 * j**2 - 1)
#             w[i] = 2 / (n - 1) * (1 - s)
#     # These weights integrate on [-1,1] to 2, so divide by 2 to get integration over [0,1].
#     w = w / 2
    
#     return u, w



# def average_squared_L2_distance_vectorized(P1, P2, x, num_cc=1000):
#     """
#     Compute the average squared L2 distance between inverse CDFs (quantile functions)
#     for corresponding columns of P1 and P2 in a fully vectorized manner.
   
#     For each column (each a PDF on grid x), we:
#       1. Normalize the PDF.
#       2. Compute its inverse CDF on a nonuniform grid u in [0,1] using Clenshaw-Curtis nodes,
#          which cluster near 0 and 1 (capturing steep behavior in the tails).
#       3. Compute the squared difference between the inverse CDFs for P1 and P2,
#          and integrate using the Clenshaw-Curtis quadrature weights.
#       4. Average the squared L2 distances over all columns.
   
#     Parameters:
#       P1, P2 : numpy arrays of shape (len(x), M), where each column is a PDF.
#       x      : 1D numpy array of grid points (assumed uniformly spaced).
#       num_cc : number of Clenshaw-Curtis nodes (default 1000).
     
#     Returns:
#       avg_sq_dist : average squared L2 distance between the inverse CDFs.
#       col_sq_dist : 1D numpy array (length M) of the squared L2 distances per column.
#     """
    
#     dx = x[1] - x[0]
#     # Get Clenshaw-Curtis nodes and weights on [0,1]
#     u_cc, w_cc = clenshaw_curtis(num_cc)  # u_cc: shape (num_cc,), w_cc: shape (num_cc,)
   
#     # Compute vectorized inverse CDFs for P1 and P2 (each shape: (num_cc, M))
#     inv_cdfs1 = compute_inverse_cdfs(P1, x, u_cc)
#     inv_cdfs2 = compute_inverse_cdfs(P2, x, u_cc)
   
#     # Compute squared difference (shape: (num_cc, M))
#     diff_sq = (inv_cdfs1 - inv_cdfs2) ** 2
#     # Integrate along the u-axis using Clenshaw-Curtis weights (broadcast weights to all columns)
#     col_sq_dist = np.sum(diff_sq * w_cc[:, np.newaxis], axis=0)
   
#     avg_sq_dist = np.sqrt(np.mean(col_sq_dist))
    
#     return avg_sq_dist