import numpy as np

from scipy import ndimage as ndi

import finufft

from fast_volume_align_ot import bh_projections

import torch
import torch_harmonics as th
# from scipy.fft import ifftn as sifftn
# import matplotlib.pyplot as plt

import spherical
import quaternionic
# import copy


### TODO
### - update variables to match paper

### - add inputs for bh_transform specifics (oversampling, max_l)
### - add the ramp filter option for the bh_transform
### - fix conventions to batch the bh_transform

### - add modes for different Fourier point spacing

### - check that odd and even grids work

### - different interpolation or smoothing method for icdf
### - vectorize the icdf step


class Transform():
    """
    NOTES:
    - in numpy convention, the 3D projection tensor is (should be):
    
        z, phi [0, 2pi)
       /
      /
     /__ __ __ x, theta [0, pi]
    |
    |
    |
    y, points

    """

    def __init__(self, f, d, omega=None, theta=None, phi=None, n_points=None, proj_type='RT', apply_ramp=False):

        self.f = f.astype(np.complex128)  # required type for nufft
        self.d = d  # dimension of input signal
        self.omega = omega
        self.theta = theta
        self.phi = phi
        self.n_points = n_points
        self.proj_type = proj_type  # options are RT = Radon transform or BH = ball harmonics (only for 3D)
        self.apply_ramp = apply_ramp
        self.shape = f.shape

        if d == 2 and len(self.shape) == 3:
            self.n, self.ny, self.nx = self.shape
        elif d == 2 and len(self.shape) == 2:
            self.n, self.ny, self.nx = 1, self.shape[0], self.shape[1]

        if d == 3 and len(self.shape) == 4:
            self.n, self.nz, self.ny, self.nx = f.shape
        elif d == 3 and len(self.shape) == 3:
            self.n, self.nz, self.ny, self.nx = 1, self.shape[0], self.shape[1], self.shape[2]

        if self.omega is None:
            self.omega = np.linspace(0, 2*np.pi, self.ny, endpoint=False)
        if self.theta is None:
            self.theta = np.linspace(0, np.pi, self.ny, endpoint=True)  # theta is [0, pi]
        if self.phi is None:
            self.phi = np.linspace(0, 2*np.pi, self.ny, endpoint=False)
        if self.n_points is None:
            self.n_points = self.ny

        self.n_omega = self.omega.size
        self.n_theta = self.theta.size
        self.n_phi = self.phi.size

        if apply_ramp:
            self.ramp = np.abs(np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True))


    def get_2d_to_1d_fourier_slice_points(self, mode=0):
        """mode: controls point spacings for 1D Fourier slices (0 = Cartesian, 1 = Chebyshev)"""

        ### initial Fourier points along y-axis
        if mode == 0:
            if self.n_points % 2 == 0:
                y = np.linspace(-1, 1, self.n_points, endpoint=False)
            else:
                y = np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True) 
        elif mode == 1:
            assert "Chebyshev points not implemented yet"
    
        ### rotate by omega (around z-axis)
        x_omega = y[:, np.newaxis] * np.sin(self.omega)[np.newaxis, :]
        y_omega = y[:, np.newaxis] * np.cos(self.omega)[np.newaxis, :]
        
        ### format for fiNUFFT
        y_omega = np.pi * y_omega.flatten()
        x_omega = np.pi * x_omega.flatten()
    
        return x_omega, y_omega


    def get_3d_to_1d_fourier_slice_points(self, mode=0):
        """mode controls point spacings for 1D Fourier slices (0 = Cartesian, 1 = Chebyshev, 2 = Fibonacci)"""
    
        ### initial Fourier points along z-axis
        if mode == 0:
            if self.n_points % 2 == 0:
                z = np.linspace(-1, 1, self.n_points, endpoint=False)
            else:
                z = np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True) 
        elif mode == 1:
            assert "Chebyshev points not implemented yet"
        elif mode == 2:
            assert "Fibonacci points not implemented yet"
        #### OM: ADDED THIS. Could we run get_3d_to_1d_fourier_slice_points in the __init__ just so z is defined and saved?
        self.z = z
        ####
        ### rotate central slice on z-axis by theta (around y-axis) to form xz-plane of 1D slices
        x_theta = z[:, np.newaxis] * np.sin(self.theta)[np.newaxis, :] 
        y_theta = np.zeros((self.n_points, self.n_theta))
        z_theta = z[:, np.newaxis] * np.cos(self.theta)[np.newaxis, :]
        
        ### rotate each slice by phi (around the z-axis)
        x_theta_phi = x_theta[np.newaxis, ...] * np.cos(self.phi[:, np.newaxis, np.newaxis])
        y_theta_phi = x_theta[np.newaxis, ...] * np.sin(self.phi[:, np.newaxis, np.newaxis])
        z_theta_phi = np.ones((self.n_phi, self.n_points, self.n_theta)) * z_theta
        
        ### format for fiNUFFT
        x_theta_phi = np.pi * x_theta_phi.flatten()
        y_theta_phi = np.pi * y_theta_phi.flatten()
        z_theta_phi = np.pi * z_theta_phi.flatten()
        
        return x_theta_phi, y_theta_phi, z_theta_phi


    def compute_nufft(self, nufft_type=2, eps=1e-8):
    
        if self.d == 2:
            plan = finufft.Plan(nufft_type, (self.nx, self.ny), self.n, eps)
            x, y = self.get_2d_to_1d_fourier_slice_points(mode=0)
            plan.setpts(x, y)
        
            f_hat = plan.execute(self.f).reshape(self.n, self.n_points, self.n_omega)
        
            if self.apply_ramp:
                f_hat *= self.ramp[:, np.newaxis]

        elif self.d == 3:
            plan = finufft.Plan(nufft_type, (self.nx, self.ny, self.nz), self.n, eps)
            x, y, z = self.get_3d_to_1d_fourier_slice_points(mode=0)
            plan.setpts(x, y, z)
        
            f_hat = plan.execute(self.f).reshape(self.n, self.n_phi, self.n_points, self.n_theta)
        
            if self.apply_ramp:
                f_hat *= self.ramp[np.newaxis, :, np.newaxis]        
        
        return f_hat


    def radon_transform(self):

        f_hat = self.compute_nufft()

        if self.d == 2:
            f_proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat, axes=1), axis=1), axes=1).real

        elif self.d == 3:
            f_proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat, axes=2), axis=2), axes=2).real
            # f_proj *= ((2 - 2/self.nz)**2) / (self.nz**2)  # EV: is nz the correct? , removed this and added dV

        return f_proj


    def ball_harmonic_projections(self, s=2):

        ### EV: note need to add input varibles for s (for angles too) and max_l
        ### also bh_projections only acts on single volume at a time
        ### could move this function outside class to simplify?
        ### scaling of the projections?
        ### option to apply the ramp filter too

        N = self.nz

        n_phi = N  ### add option to oversample
        n_theta = N  ### add option to oversample
        nz_s = s*N  

        f_proj = bh_projections.projs(self.f, N, nz_s, n_theta, n_phi, max_l=30)[np.newaxis,:,::s,:]  #EV: check newaxis here

        return f_proj.real


    # def get_cdf(self, f_proj):

    #     if self.d == 2:
    #         f_cdf = np.cumsum(f_proj, axis=1)
    #     elif self.d == 3:
    #         f_cdf = np.cumsum(f_proj, axis=2)

    #     return f_cdf
    
    
    # def normalize_cdf(self, cdfs):
    #     """normalize cdfs to maximum of 1"""

    #     if self.d == 2:
    #         s = cdfs[:, -1, :]  
    #         s = s.reshape(self.n, 1, self.n_omega)
    #     elif self.d == 3:
    #         s = cdfs[:, :, -1, :]
    #         s = s.reshape(self.n, self.n_phi, 1, self.n_theta)

    #     cdfs = cdfs / s

    #     return cdfs


    def get_cdf(self, f_proj, normalize):

        ### EV: computes the cdf by cumulative trapezoid
        ###     can probably simplify to single d_dim function

        ### EV: assuming that dz is constant for now!
        ###     need to define the grids and vol elements earlier

        ### *** the output of Radon Transform is not \int pdf = 1 ?
        ###     so I need to keep the normalize = True here for now

        dz = self.z[1]-self.z[0]
        f_cdf = np.zeros(f_proj.shape)
        
        if self.d == 2:
            avg = 0.5 * (f_proj[:, 1:, :] + f_proj[:, :-1, :])
            trap_area = avg * dz
            f_cdf[:, 1:, :] = np.cumsum(trap_area, axis=1)

            if normalize:
                f_cdf = f_cdf / f_cdf[:, -1, :]

        elif self.d == 3:
            avg = 0.5 * (f_proj[:, :, 1:, :] + f_proj[:, :, :-1, :])
            trap_area = avg * dz
            f_cdf[:, :, 1:, :] = np.cumsum(trap_area, axis=2)

            if normalize:
                f_cdf = f_cdf / f_cdf[:, :, -1, :]

        return f_cdf       
    
    
    def cdf_transform(self, normalize=True):

        if self.proj_type == 'RT':
            f_proj = self.radon_transform()
        elif self.proj_type == 'BH':
            f_proj = self.ball_harmonic_projections()
            
        f_cdf = self.get_cdf(f_proj, normalize)

        # if normalize:
        #     f_cdf = self.normalize_cdf(f_cdf)

        return f_cdf


    def signed_cdf_transform(self, normalize=True):

        if self.proj_type == 'RT':
            f_proj = self.radon_transform()
        elif self.proj_type == 'BH':
            f_proj = self.ball_harmonic_projections()
            
        f_p, f_n = hahn_decomposition(f_proj)

        f_p_cdf = self.get_cdf(f_p, normalize)
        f_n_cdf = self.get_cdf(f_n, normalize)

        # if normalize:
        #     f_p_cdf = self.normalize_cdf(f_p_cdf)
        #     f_n_cdf = self.normalize_cdf(f_n_cdf)

        return f_p_cdf, f_n_cdf


    def get_uniform_inverse_cdf(self, f_cdf):

        ### EV: to make same as paper, the grid should be defined on [-1, 1] ?, what about dt
        ###     the ball enclosing the density function ?
        ###     and so should the preceeding steps

        f_icdf = np.zeros(f_cdf.shape)
        xp = np.linspace(0, 1, self.n_points, endpoint=True)  # grid points for interpolation

        if self.d == 2:
            for n in range(self.n):
                for omega_idx in range(self.n_omega):
                    icdf = np.interp(xp, f_cdf[n, :, omega_idx], xp)
                    f_icdf[n, :, omega_idx] = icdf
            f_icdf[:, 0, :] = 0  # EV: ENFORCE BOUNDS (not sure why this helps)
            f_icdf[:, -1, :] = 1

        if self.d == 3:
            for n in range(self.n):
                for phi_idx in range(self.n_phi):
                    for theta_idx in range(self.n_theta):
                        f_icdf[n, phi_idx, :, theta_idx] = np.interp(xp, f_cdf[n, phi_idx, :, theta_idx], xp)
            f_icdf[:, :, 0, :] = 0  # EV: ENFORCE BOUNDS (not sure why this helps)
            f_icdf[:, :, -1, :] = 1

        return f_icdf
        

    def inverse_cdf_transform(self, normalize=True):

        f_cdf = self.cdf_transform(normalize=normalize)
        f_icdf = self.get_uniform_inverse_cdf(f_cdf)

        return f_icdf


    def signed_inverse_cdf_transform(self, normalize=True):

        f_p_cdf, f_n_cdf = self.signed_cdf_transform(normalize=normalize)
        f_p_icdf = self.get_uniform_inverse_cdf(f_p_cdf)
        f_n_icdf = self.get_uniform_inverse_cdf(f_n_cdf)

        return f_p_icdf, f_n_icdf


def hahn_decomposition(f):
    """takes signed array f -> (f+, f-), both of size f"""

    f_p = np.where(f >= 0, f, 0)
    f_n = np.where(f <= 0, f, 0)

    return f_p, f_n


def translate_2d(array, ty, tx):

    return np.roll(array, (ty, tx), axis=(0,1))


def translate_3d(array, tz, ty, tx):

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


def rotation_matrix_from_zyz(omega, theta, phi):

    Rz1 = rotate_z(omega)
    Ry  = rotate_y(theta)
    Rz2 = rotate_z(phi)

    R = Rz2 @ Ry @ Rz1

    return R


def rotate_voxel_zyz(voxels, omega, theta, phi, order=1):
    """
    Extrinsic ZYZ rotation preserving a right-handed spatial interpretation.
    """
    # Base rotation (about fixed axes)
    # Rz1 = rotate_z(omega)
    # Ry = rotate_y(theta)
    # Rz2 = rotate_z(phi)
    # R_inv = Rz2 @ Ry @ Rz1

    # OM: updates.
    #For an explanation of the minus signs,
    #our rotation R = Rz1@Ry@Rz2 should in the
    #convention in the overleaf be the rotation
    #mapping the vector with spherical coordinates
    #(theta, varphi) to the z-axis and this
    #is only true if we use minus signs. If not,
    #it sends the vector with spherical coordinates
    #(-theta, -varphi) to the z-axis.
    Rz1 = rotate_z(-omega)
    Ry = rotate_y(-theta)
    Rz2 = rotate_z(-phi)
    R_inv = (Rz1 @ Ry @ Rz2).T 
    #We want to rotate a function f(x,y,z)
    #by sending it to f(R_inv*(x,y,z)) = f(R^T*(x,y,z))
    #so need the transpose in affine_transform

    center = np.array(voxels.shape) / 2
    offset = center - R_inv @ center

    rotated = ndi.affine_transform(
        voxels,
        R_inv,
        offset=offset,
        order=order,
        mode='constant',
        cval=0.0
    )
    return rotated 


def rotate_voxel_zyz_fourier(voxels, omega, theta, phi):
    """
    Extrinsic ZYZ rotation preserving a right-handed spatial interpretation.
    """
    Rz1 = rotate_z(-omega)
    Ry = rotate_y(-theta)
    Rz2 = rotate_z(-phi)
    R_inv = (Rz1 @ Ry @ Rz2).T

    n = voxels.shape[0]
    start = -n // 2 + 1
    end = n // 2

    if n % 2 == 0:
        start -= 1
        end -= 1
        
    start /= n / 2 #/np.pi
    end /= n / 2 #/np.pi

    grid = slice(start, end, n*1j)
    # grid = slice(start, end, n)

    z, y, x = np.mgrid[grid, grid, grid].astype(np.float64)
    # x,y,z = np.mgrid[grid, grid, grid].astype(np.float64)


    pts = np.pi * np.vstack([np.real(x).flatten(),np.real(y).flatten(),np.real(z).flatten(),])
    pts_rot = R_inv @ pts

    pts_rot = pts_rot.reshape(3, -1)[::-1]
        
    fourier_pts = np.ascontiguousarray(
        np.mod(pts_rot + np.pi, 2 * np.pi) - np.pi
    )

    num_pts = pts_rot.shape[1]

    epsilon = 1e-14
    transform_plan = finufft.Plan(
        nufft_type=2,
        n_modes_or_dim=(n, n, n),
        eps=epsilon,
        n_trans=1,
        dtype=np.complex128,
        upsampfac=2,
    )
    transform_plan.setpts(pts_rot[2,:], pts_rot[1,:], pts_rot[0,:])

    signal = np.asarray(voxels, dtype=np.complex128, order="C")

    result = transform_plan.execute(signal).reshape(n,n,n)

    result = np.fft.ifftshift(result)
    result = np.fft.ifftn(result)
    rotated = np.transpose(np.real(np.fft.fftshift(result)).reshape(n,n,n),(2,1,0))
    
    return rotated 


def rotate_voxels_3d(voxels, R, order=1):

    center = np.array(voxels.shape) / 2
    offset = center - R @ center
    rotated = ndi.affine_transform(
        voxels,
        R,
        offset=offset,
        order=order,
        mode='constant',
        cval=0.0
    )    
    return rotated


def rotmat_to_euler_zyz(R):
    alpha = np.arctan2(R[1,2],R[0,2])
    beta = np.arctan2( np.sqrt(1 - R[2,2]**2),R[2,2] )
    omega = np.arctan2( R[2,1], -R[2,0] )
    
    return alpha, beta, omega


# def fast_block_average_downsample(x, factor):
#     N = x.shape[0]
#     n = N // factor
    
#     blocks = x.reshape(n, factor, n, factor, n, factor)
#     return np.einsum('abcdef->ace', blocks) / factor**3

    
def block_average_downsample(x, factor):

    ### EV: block average 3D voxel array
    ###     simple downsample, preserves sign and mass
    
    idx = np.arange(0, x.shape[0], factor)

    sum_x = np.add.reduceat(x, idx, axis=0)
    sum_xy = np.add.reduceat(sum_x, idx, axis=1)
    sum_xyz = np.add.reduceat(sum_xy, idx, axis=2)

    return sum_xyz / factor**3


def fourier_downsample(array, factor=1, rescale=False):
    """Downsample array by cropping its Fourier transform (factor 2 would give 100pix -> 50pix)"""

    ### EV: does not apply an ideal low-pass filter first
    ###     mass and sign not preserved in this version
    ###     trade-off between simple downsampling and low-freq preservation / aliasing compared to block version
    
    assert factor >= 1, "scale factor must be greater than 1"
    
    shape = array.shape
    center = [d//2 for d in shape]
    new_shape = [int(d / factor) for d in shape]
    
    F = np.fft.fftshift(np.fft.fftn(array))
    idx = tuple([slice(center[i] - new_shape[i]//2, center[i] + new_shape[i]//2) for i in range(len(shape))])
    F = F[idx] 
    
    if rescale:
        F = F * (np.prod(new_shape) / np.prod(shape))
    
    f_downsample = np.fft.ifftn(np.fft.ifftshift((F)))
    
    return f_downsample.real


def cumulative_trapezoid(f, z, normalize=True):
    """returns cumulative integral with cdf[0] = 0"""

    dz = z[1:] - z[:-1]
    avg = 0.5 * (f[1:] + f[:-1])
    trap_area = avg * dz
    cdf = np.concatenate([[0], np.cumsum(trap_area)])

    if normalize:
        cdf = cdf / cdf[-1]

    return cdf