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
            f_proj *= ((2 - 2/self.nz)**2) / (self.nz**2)  # EV: is nz the correct variable here?

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


    def get_cdf(self, f_proj):

        if self.d == 2:
            f_cdf = np.cumsum(f_proj, axis=1)
        elif self.d == 3:
            f_cdf = np.cumsum(f_proj, axis=2)

        return f_cdf
    
    
    def normalize_cdf(self, cdfs):
        """normalize cdfs to maximum of 1"""

        if self.d == 2:
            s = cdfs[:, -1, :]  
            s = s.reshape(self.n, 1, self.n_omega)
        elif self.d == 3:
            s = cdfs[:, :, -1, :]
            s = s.reshape(self.n, self.n_phi, 1, self.n_theta)

        cdfs = cdfs / s

        return cdfs
    
    
    def cdf_transform(self, normalize=True):

        if self.proj_type == 'RT':
            f_proj = self.radon_transform()
        elif self.proj_type == 'BH':
            f_proj = self.ball_harmonic_projections()
            
        f_cdf = self.get_cdf(f_proj)

        if normalize:
            f_cdf = self.normalize_cdf(f_cdf)

        return f_cdf


    def signed_cdf_transform(self, normalize=True):

        if self.proj_type == 'RT':
            f_proj = self.radon_transform()
        elif self.proj_type == 'BH':
            f_proj = self.ball_harmonic_projections()
            
        f_p, f_n = hahn_decomposition(f_proj)

        f_p_cdf = self.get_cdf(f_p)
        f_n_cdf = self.get_cdf(f_n)

        if normalize:
            f_p_cdf = self.normalize_cdf(f_p_cdf)
            f_n_cdf = self.normalize_cdf(f_n_cdf)

        return f_p_cdf, f_n_cdf


    def get_uniform_inverse_cdf(self, f_cdf):

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

        
    # def align(self, f, g, fle):
    #     #icf and icg are the icdfs of the projection tensors Pf and Pg
    #     #fle is a basis object from FLE3D, needed to call some functions that massage
    #     #the input to the fast spherical harmonics transform. We can get around using this
    #     #by copying some functions from FLE3D, if we want.
    #     #Step 1: replace icf and icg by 1/abs(z)*icf and 1/abs(z)*icg
    #     #maybe we could make this z into a self.z in the __init__?

    #     ## the following two lines are probably a waste,
    #     ## but I didn't know how to do it better
    #     f = copy.deepcopy(f)
    #     g = copy.deepcopy(g)
    #     ##
    #     f = np.squeeze(f)
    #     g = np.squeeze(g)
        
    #     lmax = fle.lmax+1
        
    #     if self.n_points % 2 == 0:
    #         z = np.linspace(-1, 1, self.n_points, endpoint=False)
    #     else:
    #         z = np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True) 
    #     for i in range(f.shape[1]):
    #         #the entry for z = 0 is not rotated by the alignment so its value does not matter
    #         #(unless we use a spherical harmonics transform that does some interpolation)
    #         #this 1/np.abs(z[i]) is actually cancelled by something below, so we should maybe remove it
    #         #here and below.
    #         if z[i] != 0:
    #             f[:,i,:] = f[:,i,:] / (np.abs(z[i]))
    #             g[:,i,:] = g[:,i,:] / (np.abs(z[i]))
            
    #     #Step 2: compute spherical harmonics coefficients
    #     sht = th.RealSHT(self.n_theta, self.n_phi, lmax=lmax, grid="equiangular", csphase=True).to("cpu")

    #     #torch_harmonics wants the order radius, theta, phi
    #     f_swapped = np.transpose(f,(1,2,0))
    #     g_swapped = np.transpose(g,(1,2,0))


    #     #Question: icf and icg are real-valued, right? Torch-harmonics only likes real inputs
    #     #If not, we just have to call it twice, using the imaginary parts for the second call.
    #     ### EV: yes, real-valued. but we might want option for imaginary to compute Euclidean dist from the FT
    #     fhat = fle.torch_reshape_order_t(sht(torch.DoubleTensor(np.real(f_swapped)).to("cpu")).cpu().numpy())
    #     ghat = fle.torch_reshape_order_t(sht(torch.DoubleTensor(np.real(g_swapped)).to("cpu")).cpu().numpy())
    #     fhat = fhat[:,:,:(2*lmax+1)]
    #     ghat = ghat[:,:,:(2*lmax+1)]
    #     #these are now in order radius, l, m, where for fixed radius and l, the m index varies like
    #     #m=0,-1,1,-2,2,-3,3, ...

        
    #     #Step 3: compute the integral
    #     #the d^\ell_{mh} are non-zero only if |m| <= \ell, |h| <= \elll
    #     #so we can zero-pad them and make tensors of size lmax*(2*lmax+1)*(2*lmax+1)
    #     #and then just take the sum over the first dimension of d^\ell_{mh} * d^\ell_{hm'}*I^\ell_{mm'}
    #     #and store this in T(m,h,m')

    #     Ilm = np.zeros((lmax,2*lmax+1,2*lmax+1),dtype=np.complex128)
    #     #we need to remember to multiply by r^2, which cancels out the 1/abs(z) above...
    #     for m1 in range(-lmax, lmax+1):
    #         m1ind = 2*np.abs(m1) - (m1 < 0)
    #         for m2 in range(-lmax, lmax+1):
    #             m2ind = 2*np.abs(m2) - (m2 < 0)
    #             #wfhat, ghat have a different ordering. m1ind and m2ind fix this
    #             for i in range(fhat.shape[0]):
    #                 #should this actually only be for positive z? also need to divide by delta z...
    #                 #we can use better quadrature here to get increased accuracy
    #                 if z[i] > 0:
    #                     Ilm[:,m1+lmax,m2+lmax] += fhat[i,:,m1ind]*np.conj(ghat[i,:,m2ind])*np.abs(z[i])**2

    #     wigner = spherical.Wigner(lmax)
    #     R = quaternionic.array.from_euler_angles(0, np.pi/2, 0)
    #     d = wigner.D(R)

    #     dlm = np.zeros((lmax, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    #     for m1 in range(-lmax, lmax+1):
    #         for m2 in range(-lmax, lmax+1):
    #             for l in range(lmax):
    #                 if abs(m1) <= l and abs(m2) <= l:
    #                     # dlm[l,m1+lmax,m2+lmax] = d[wigner.Dindex(l, m1, m2)]
    #                     dlm[l,m1+lmax,m2+lmax] = d[wigner.Dindex(l, m2, m1)]
    #             # dlm[:,m1+lmax,m2+lmax] = wigner_dl(0, lmax, m1, m2, np.pi/2)
    #             #computes all wigner d-matrices for l between 0 and lmax at fixed m1,m2 and angle pi/2

    #     T = np.zeros((2*lmax+1, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    #     for m in range(-lmax, lmax+1):
    #         for h in range(-lmax, lmax+1):
    #             for mp in range(-lmax, lmax+1):
    #                 T[m+lmax,h+lmax,mp+lmax] = np.sum(dlm[:,m+lmax,h+lmax]*dlm[:,h+lmax,mp+lmax]*Ilm[:,m+lmax,mp+lmax])
                    
    #     #Step 4: do an IFFT
    #     ###### c = np.fft.fftshift(sifftn(np.fft.ifftshift(T), workers=-1))
    #     #TODO: WHICH ONE OF THESE IS FASTER??
    #     c = np.fft.ifftn(np.fft.ifftshift(T))
    #     # c = sifftn(np.fft.ifftshift(T), workers=-1)
    #     # c = sifftn(np.fft.ifftshift(T), workers=1)

    #     xiind,etaind,omegaind = np.unravel_index(np.argmax(c), c.shape)
        
    #     #xi, eta, omega need to be changed from indices to values
    #     xi = 2*np.pi*xiind/c.shape[0]
    #     eta = 2*np.pi*etaind/c.shape[1]
    #     omega = 2*np.pi*omegaind/c.shape[2]
        
    #     #the below phi, theta, psi is what the paper says
    #     # phi = xi + np.pi/2
    #     # theta = np.pi - eta
    #     # psi = omega + np.pi/2
        
    #     # but the below phi, theta, psi are what work.
    #     # Not sure why the minus signs are needed...
    #     phi = -(xi + np.pi/2)
    #     theta = np.pi - eta
    #     psi = -(omega + np.pi/2)

    #     R = rotate_zyz(phi,theta,psi)
        
    #     return R


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


def rotate_zyz(omega, theta, phi):

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


### EV: add convention so that the angles are always in certain frame , i.e. no +- 360 deg
###     ***** below chatgpt function is close but not quite right *****
###     but I know Oscar can fix it :)  , also wasn't sure with paper conventions

### EV: actually maybe it's way easier to just use scipy rotate class somehow.
###     e.g. Rotation.from_matrix(c).as_euler('zyz', degrees=True)

def rotmat_to_euler_zyz(R, tol=1e-8):
    """
    Convert a 3x3 rotation matrix to intrinsic ZYZ Euler angles.

    Returns:
        alpha in [0, 2π)
        beta  in [0, π/2]
        gamma in [0, 2π)

    Canonical choices:
        - prefer beta <= π/2 to remove ±π ambiguity
        - handle beta = 0 and beta = π singularities
    """

    def wrap_2pi(x):
        return np.mod(x, 2*np.pi)

    # --- beta ---
    r33 = np.clip(R[2, 2], -1.0, 1.0)
    beta = np.arccos(r33)

    # --- general case ---
    if abs(np.sin(beta)) > tol:
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
    else:
        # --- singular cases ---
        if beta < tol:
            beta = 0.0
            alpha = np.arctan2(R[1, 0], R[0, 0])
            gamma = 0.0
        else:
            beta = np.pi
            alpha = np.arctan2(-R[1, 0], -R[0, 0])
            gamma = 0.0

    # --- enforce canonical beta <= pi/2 ---
    if beta > np.pi / 2:
        alpha += np.pi
        beta = np.pi - beta
        gamma += np.pi

    # --- wrap to positive ranges ---
    alpha = wrap_2pi(alpha)
    gamma = wrap_2pi(gamma)

    return gamma, beta, alpha