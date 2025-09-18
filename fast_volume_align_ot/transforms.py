import numpy as np

import finufft

### EV: TODO
### - make class to compute desired transform in one line
### - add ramp filter functions

def compute_projection_tensor_from_voxel_array(f):
    ### TODO
    ### - add input arguments, simplify, move inside transform class
    ### - add option to make different sampling points (Fibonacci, Chebyshev, etc)
    ### - update variables to match paper

    nz, ny, nx = f.shape
    
    ### EV: add option to change these if desired
    n_points = nz  # number of Fourier grid points per 1D slice, scales with volume size
    n_phi = nz  # number of projection angles in phi
    n_theta = nz  # number of projection angles in theta
    
    ### start with a central slice along the z-axis
    ### rotate by theta (around y-axis) to form xz-plane of 1D slices
    
    ### EV: TODO: define point spacing based on odd or even grid size (DFT conventions)
    x3 = np.linspace(-(1 - (1/n_points)), 1 - (1/n_points), n_points, endpoint=True)  # initial Fourier points along z-axis (x3)
    # x3 = np.linspace(-1, 1, n_points, endpoint=False)  # initial Fourier points along z-axis (x3)
    # x3 = np.linspace(1 - (1/n_points), -(1 - (1/n_points)), n_points, endpoint=True)
    
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    x1_theta = x3[:, np.newaxis] * np.sin(theta)[np.newaxis, :]
    x2_theta = np.zeros((n_points, n_theta))
    x3_theta = x3[:, np.newaxis] * np.cos(theta)[np.newaxis, :]
    
    ### now rotate each slice by phi (around the z-axis)
    
    phi = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    x1_theta_phi = x1_theta[np.newaxis, ...] * np.cos(phi[:, np.newaxis, np.newaxis])
    x2_theta_phi = x1_theta[np.newaxis, ...] * np.sin(phi[:, np.newaxis, np.newaxis])
    x3_theta_phi = np.ones((n_phi, n_points, n_theta)) * x3_theta
    
    ### format for fiNUFFT
    
    x1_theta_phi = np.pi * x1_theta_phi.flatten()
    x2_theta_phi = np.pi * x2_theta_phi.flatten()
    x3_theta_phi = np.pi * x3_theta_phi.flatten()

    nufft_type=2
    eps=1e-8
    n_transforms=1
    
    plan = finufft.Plan(nufft_type, (nx, ny, nz), n_transforms, eps)
    
    ### EV: not sure what correct ordering is to match conventions
    plan.setpts(x1_theta_phi, x2_theta_phi, x3_theta_phi)
    
    f_hat_slices = plan.execute(f.astype(np.complex128))
    f_hat_slices = f_hat_slices.reshape((n_phi, n_points, n_theta))
    
    f_projection_tensor = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat_slices, axes=1), axis=1), axes=1).real

    return f_projection_tensor


def pdf_tensor_to_cdf_tensor(F):
   
    ### EV: will need to add a normalization step for ramp filtered slices

    return np.cumsum(F, axis=1)


def cdf_tensor_to_icdf_tensor(C):
    
    U = np.zeros(C.shape)
    n_phi, nx, n_theta = C.shape
     
    ### EV: add option later to change number of points and spacing if desired
    x = np.linspace(0, 1, nx, endpoint=True)
    
    ### EV: NEED TO VECTORIZE THIS
    for phi in range(n_phi):
        for theta in range(n_theta):
            U[phi, :, theta] = np.interp(x, C[phi, :, theta], x)
    
    ### EV: ENFORCE BOUNDS (not sure why this helps)
    U[:, 0, :] = 0
    U[:, -1, :] = 1

    return U


def translate(array, tz, ty, tx):

    return np.roll(array, (tz, ty, tx), axis=(0,1,2))


def rotate_x(theta):

    return np.array([[1,       0,              0      ],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])


def rotate_y(theta):

    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [       0,       1,        0     ],
                     [-np.sin(theta), 0, np.cos(theta)]])


def rotate_z(theta):

    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [      0,              0,       1]])