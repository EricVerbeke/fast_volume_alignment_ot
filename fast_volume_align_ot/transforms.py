import numpy as np

import finufft

### TODO
### - update variables to match paper

class Transform():
    """
    TODO:
    - add modes for different Fourier point spacing
    - different interpolation or smoothing method for icdf
    - vectorize the icdf step
    - allow for odd or even signal size
    """

    def __init__(self, f, d, omega=None, theta=None, phi=None, n_points=None, apply_ramp=False):

        self.f = f.astype(np.complex128)  # required type for nufft
        self.d = d  # dimension of input signal
        self.omega = omega
        self.theta = theta
        self.phi = phi
        self.n_points = n_points
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
            self.theta = np.linspace(0, 2*np.pi, self.ny, endpoint=False)
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
            z = np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True)
        elif mode == 1:
            assert "Chebyshev points not implemented yet"
        elif mode == 2:
            assert "Fibonacci points not implemented yet"
    
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
            f_rt = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat, axes=1), axis=1), axes=1).real

        elif self.d == 3:
            f_rt = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat, axes=2), axis=2), axes=2).real

        return f_rt


    def get_cdf(self, f_rt):

        if self.d == 2:
            f_cdf = np.cumsum(f_rt, axis=1)
        elif self.d == 3:
            f_cdf = np.cumsum(f_rt, axis=2)

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

        f_rt = self.radon_transform()
        f_cdf = self.get_cdf(f_rt)

        if normalize:
            f_cdf = self.normalize_cdf(f_cdf)

        return f_cdf


    def signed_cdf_transform(self, normalize=True):

        f_rt = self.radon_transform()
        f_p, f_n = hahn_decomposition(f_rt)

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