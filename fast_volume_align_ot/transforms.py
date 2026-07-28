import numpy as np

import finufft

from scipy import ndimage as ndi

import torch
import torch_harmonics as th

import spherical
import quaternionic

from fast_volume_align_ot import bh_projections


class Grids():
    """
    Define the grids to be used for the Transform class.
    Grids can be input manually, or use default.
    Default settings: grid spacing is uniform, includes endpoints,
    and the [theta, phi, t] grids are not oversampled
    """

    ### EV: 
    ### - do we want to control for odd/even grid size (and when oversampling)?
    ### - currently our NUDFT doesn't support non-uniform space points (i.e. a Chebyshev grid)
    ### - need to confirm boundaries of grids

    def __init__(self, 
                 N,
                 oversample_grids=['thetas', 'phis', 'ts'],
                 oversample_factor=1,
                 omegas=None, 
                 thetas=None, 
                 phis=None, 
                 zs=None, 
                 ss=None,
                 ts=None, 
                 z_endpoint=True,
                 s_endpoint=True,
                 t_endpoint=True,
                 z_grid_type='uniform',
                 s_grid_type='uniform',
                 t_grid_type='uniform',
                ):

        self.N = N
        self.k = oversample_factor
        self.omegas = omegas
        self.thetas = thetas
        self.phis = phis
        self.zs = zs
        self.ss = ss
        self.ts = ts
        self.z_endpoint = z_endpoint
        self.s_endpoint = s_endpoint
        self.t_endpoint = t_endpoint
        self.z_grid_type = z_grid_type
        self.s_grid_type = s_grid_type
        self.t_grid_type = t_grid_type

        self.n_omegas = self.N
        self.n_thetas = self.N
        self.n_phis = self.N
        self.n_zs = self.N
        self.n_ss = self.N
        self.n_ts = self.N

        for grid in oversample_grids:
            if grid == 'omegas':
                self.n_omegas = np.round(self.k*self.N).astype('int')
            elif grid == 'thetas':
                self.n_thetas = np.round(self.k*self.N).astype('int')
            elif grid == 'phis':
                self.n_phis = np.round(self.k*self.N).astype('int')
            elif grid == 'zs':
                self.n_zs = np.round(self.k*self.N).astype('int')
            elif grid == 'ss':
                self.n_ss = np.round(self.k*self.N).astype('int')
            elif grid == 'ts':
                self.n_ts = np.round(self.k*self.N).astype('int')
        
        if self.omegas is None:
            self.omegas = np.linspace(0, 2*np.pi, self.n_omegas, endpoint=False)
        if self.thetas is None:
            self.thetas = np.linspace(0, np.pi, self.n_thetas, endpoint=True)  # theta is [0, pi]
        if self.phis is None:
            self.phis = np.linspace(0, 2*np.pi, self.n_phis, endpoint=False)  # EV: endoint=True?

        if self.zs is None:
            if z_grid_type == 'uniform':
                if self.n_zs % 2 == 0:
                    self.zs = np.linspace(-1, 1, self.n_zs, endpoint=False)
                else:
                    self.zs = np.linspace(-(1 - (1/self.n_zs)), 1 - (1/self.n_zs), self.n_zs, endpoint=True) 

            elif z_grid_type == 'chebyshev':
                if z_endpoint:
                    self.zs = chebyshev_nodes(self.n_zs, a=-1.0, b=1.0, endpoint=True)
                else:
                    self.zs = chebyshev_nodes(self.n_zs, a=-1.0, b=1.0, endpoint=False)

        # if self.ss is None:
        #     if s_grid_type == 'uniform':
        #         if self.n_ss % 2 == 0:
        #             self.ss = np.linspace(-1, 1, self.n_ss, endpoint=False)
        #         else:
        #             self.ss = np.linspace(-(1 - (1/self.n_ss)), 1 - (1/self.n_ss), self.n_ss, endpoint=True) 

        if self.ts is None:
            if t_grid_type == 'uniform':
                self.ts = np.linspace(0, 1, self.n_ts, endpoint=True)
            
            elif t_grid_type == 'chebyshev':
                if t_endpoint == True:
                    self.ts = chebyshev_nodes(self.n_ts, a=0, b=1.0, endpoint=True)
                else:
                    self.ts = chebyshev_nodes(self.n_ts, a=0, b=1.0, endpoint=False)
            elif t_grid_type == 'gauss-legendre':
                xg, wg = np.polynomial.legendre.leggauss(self.n_ts)
                self.ts = 1/2*xg + 1/2
                self.wg = 1/2*wg

    
class Transform():
    """
    in numpy convention, the 3D projection tensor is:
    
        z, phi [0, 2pi)
       /
      /
     /__ __ __ x, theta [0, pi]
    |
    |
    |
    y, points

    """

    def __init__(self, f, d, grid, projection_method='RT', apply_ramp=False):

        self.f = f.astype(np.complex128)  # required type for nufft
        self.d = d  # dimension of input signal
        self.grid = grid
        self.projection_method = projection_method  # options are RT = Radon transform or BH = ball harmonics (only for 3D)
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


    def get_2d_to_1d_fourier_slice_points(self):
    
        ### rotate by omega (around z-axis)
        x_phi = self.grid.zs[:, np.newaxis] * np.sin(self.grid.phis)[np.newaxis, :]
        y_phi = self.grid.zs[:, np.newaxis] * np.cos(self.grid.phis)[np.newaxis, :]
        
        ### format for fiNUFFT
        y_omega = np.pi * y_omega.flatten()
        x_omega = np.pi * x_omega.flatten()
    
        return x_omega, y_omega


    def get_3d_to_1d_fourier_slice_points(self):

        ### rotate central slice on z-axis by theta (around y-axis) to form xz-plane of 1D slices
        x_theta = self.grid.zs[:, np.newaxis] * np.sin(self.grid.thetas)[np.newaxis, :] 
        y_theta = np.zeros((self.grid.n_zs, self.grid.n_thetas))
        z_theta = self.grid.zs[:, np.newaxis] * np.cos(self.grid.thetas)[np.newaxis, :]
        
        ### rotate each slice by phi (around the z-axis)
        x_theta_phi = x_theta[np.newaxis, ...] * np.cos(self.grid.phis[:, np.newaxis, np.newaxis])
        y_theta_phi = x_theta[np.newaxis, ...] * np.sin(self.grid.phis[:, np.newaxis, np.newaxis])
        z_theta_phi = np.ones((self.grid.n_phis, self.grid.n_zs, self.grid.n_thetas)) * z_theta
        
        ### format for fiNUFFT
        x_theta_phi = np.pi * x_theta_phi.flatten()
        y_theta_phi = np.pi * y_theta_phi.flatten()
        z_theta_phi = np.pi * z_theta_phi.flatten()
        
        return x_theta_phi, y_theta_phi, z_theta_phi


    def compute_nufft(self, nufft_type=2, eps=1e-8):
    
        if self.d == 2:
            plan = finufft.Plan(nufft_type, (self.nx, self.ny), self.n, eps)
            x, y = self.get_2d_to_1d_fourier_slice_points()
            plan.setpts(x, y)
        
            f_hat = plan.execute(self.f).reshape(self.n, self.grid.n_zs, self.grid.n_omegas)
        
            if self.apply_ramp:
                ramp = np.abs(self.grid.zs[:, np.newaxis])
                f_hat = f_hat * ramp

        elif self.d == 3:
            plan = finufft.Plan(nufft_type, (self.nx, self.ny, self.nz), self.n, eps)
            x, y, z = self.get_3d_to_1d_fourier_slice_points()
            plan.setpts(x, y, z)
        
            f_hat = plan.execute(self.f).reshape(self.n, self.grid.n_phis, self.grid.n_zs, self.grid.n_thetas)
        
            if self.apply_ramp:
                ramp = np.abs(self.grid.zs[np.newaxis, :, np.newaxis])
                f_hat = f_hat * ramp
        
        return f_hat


    def radon_transform(self):

        ### EV: will need to change this function if using Chebyshev (or other) points 

        f_hat = self.compute_nufft()

        if self.d == 2:
            f_proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat, axes=1), axis=1), axes=1).real

        elif self.d == 3:

            f_proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat, axes=2), axis=2), axes=2).real
            ### we are integrating over dx dy, this assumes uniform grid
            f_proj = f_proj * (self.grid.zs[1] - self.grid.zs[0])**2

        return f_proj


    def ball_harmonic_projections(self, s=2):

        ### EV: THIS NEEDS TO BE UPDATED WITH NEW GRID CLASS
        ### - need to add input varibles for s (for angles too) and max_l
        ### - scaling of the projections?
        ### - add inputs for bh_transform specifics (oversampling, max_l)
        ### - add the ramp filter option for the bh_transform
        ### - fix conventions to vectorize the bh_transform

        ### EV: these will be replaced with grid class
        N = self.nz
        n_phi = N
        n_theta = N
        nz_s = s*N  

        f_proj = bh_projections.projs(self.f, N, nz_s, n_theta, n_phi, max_l=30)[np.newaxis,:,::s,:]  #EV: check newaxis here

        return f_proj.real


    def get_cdf(self, f_proj, normalize):
        
        dz = np.diff(self.grid.zs)
        
        if self.d == 2:
            pass
        #     f_cdf = np.zeros((self.n, self.grid.n_phis, self.grid.n_zs))
        #     avg = 0.5 * (f_proj[:, 1:, :] + f_proj[:, :-1, :])
        #     trap_area = avg * dz[np.newaxis, :, np.newaxis]
        #     f_cdf[:, 1:, :] = np.cumsum(trap_area, axis=1)

        #     if normalize:
        #         s = f_cdf[:, -1, :]
        #         f_cdf = f_cdf / s.reshape(self.n, 1, self.grid.n_phis)

        elif self.d == 3:
            ### compute the cumulative trapezoid
            f_cdf = np.zeros((self.n, self.grid.n_phis, self.grid.n_zs, self.grid.n_thetas))
            avg = 0.5 * (f_proj[:, :, 1:, :] + f_proj[:, :, :-1, :])
            trap_area = avg * dz[np.newaxis, np.newaxis, :, np.newaxis]
            f_cdf[:, :, 1:, :] = np.cumsum(trap_area, axis=2)
            
            if normalize:
                s = f_cdf[:, :, -1, :]
                f_cdf = f_cdf / s.reshape(self.n, self.grid.n_phis, 1, self.grid.n_thetas)

        return f_cdf       
    
    
    def cdf_transform(self, normalize=False):

        if self.projection_method == 'RT':
            f_proj = self.radon_transform()
            
        elif self.projection_method == 'BH':
            f_proj = self.ball_harmonic_projections()
            
        f_cdf = self.get_cdf(f_proj, normalize)

        return f_cdf


    def signed_cdf_transform(self, normalize=False):

        if self.projection_method == 'RT':
            f_proj = self.radon_transform()
            
        elif self.projection_method == 'BH':
            f_proj = self.ball_harmonic_projections()
            
        f_p, f_n = hahn_decomposition(f_proj)

        f_p_cdf = self.get_cdf(f_p, normalize)
        f_n_cdf = np.abs(self.get_cdf(f_n, normalize))  # EV: if normalize=False , I think I need to have absolute value here

        return f_p_cdf, f_n_cdf


    def get_inverse_cdf(self, f_cdf):

        ### TODO: vectorize the icdf step, fix 2D version
        
        if self.d ==2:
            pass
        #     f_icdf = np.zeros((self.n, self.grid.n_ts, self.grid.n_phis))

        #     for n in range(self.n):
        #         for phi_idx in range(self.grid.n_phis):
        #             icdf = np.interp(self.grid.ts, f_cdf[n, :, phi_idx], self.grid.zs)
        #             f_icdf[n, :, phi_idx] = icdf
            
        #     if self.grid.t_endpoint == True:
        #         f_icdf[:, 0, :] = -1  # EV: ENFORCE BOUNDS (not sure why this helps)
        #         f_icdf[:, -1, :] = 1
        
        
        if self.d == 3:
            f_icdf = np.zeros((self.n, self.grid.n_phis, self.grid.n_ts, self.grid.n_thetas))

            for n in range(self.n):
                for phi_idx in range(self.grid.n_phis):
                    for theta_idx in range(self.grid.n_thetas):
                        
                        cdf = f_cdf[n, phi_idx, :, theta_idx]
                        
                        if self.grid.t_grid_type == 'uniform':
                            ts = np.linspace(0, cdf[-1], self.grid.n_ts, endpoint=True)
                        elif self.grid.t_grid_type == 'chebyshev':
                            ts = chebyshev_nodes(self.grid.n_ts, a=0, b=cdf[-1], endpoint=True)
                        elif self.grid.t_grid_type == 'gauss-legendre':
                            ts = self.grid.ts*cdf[-1]

                        icdf = np.interp(ts, cdf, self.grid.zs)
                        f_icdf[n, phi_idx, :, theta_idx] = icdf

        return f_icdf
        

    def inverse_cdf_transform(self, normalize=False):

        f_cdf = self.cdf_transform(normalize)
        f_icdf = self.get_inverse_cdf(f_cdf)

        return f_icdf


    def signed_inverse_cdf_transform(self, normalize=False):

        f_p_cdf, f_n_cdf = self.signed_cdf_transform(normalize)
        f_p_icdf = self.get_inverse_cdf(f_p_cdf)
        f_n_icdf = self.get_inverse_cdf(f_n_cdf)

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


def rotate_voxel_zyz_fourier(voxels, omega, theta, phi):
    """Extrinsic ZYZ rotation preserving a right-handed spatial interpretation"""
    
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

    z, y, x = np.mgrid[grid, grid, grid].astype(np.float64)

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

    
def block_average_downsample(x, factor):

    ### EV: block average 3D voxel array. simple downsample, preserves sign and mass
    
    idx = np.arange(0, x.shape[0], factor)

    sum_x = np.add.reduceat(x, idx, axis=0)
    sum_xy = np.add.reduceat(sum_x, idx, axis=1)
    sum_xyz = np.add.reduceat(sum_xy, idx, axis=2)

    return sum_xyz / factor**3


def fourier_downsample(array, factor=1, rescale=False):
    """Downsample array by cropping its Fourier transform (factor 2 would give 100pix -> 50pix)"""

    ### EV: - does not apply an ideal low-pass filter first
    ###     - mass and sign not preserved in this version
    ###     - trade-off between simple downsampling and low-freq preservation / 
    ###       aliasing compared to block version
    
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


def chebyshev_nodes(n, a=-1.0, b=1.0, endpoint=True):
    """ Generate n Chebyshev nodes on [-1, 1] or [0, 1]. """

    k = np.arange(n)[::-1]

    if endpoint:
        x = np.cos(np.pi * k / (n - 1))
    else:
        x = np.cos((2*k + 1) * np.pi / (2*n))
        
    x = 0.5 * (a + b) + 0.5 * (b - a) * x
    
    return x