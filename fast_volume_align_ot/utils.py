import pickle
import numpy as np

from scipy import ndimage as ndi


# def zero_pad_voxel_array_to_size(f, N):
#     pass


def rotate_image(array, degrees, reshape=False):
    
    return ndi.rotate(array, degrees, reshape=reshape)


def radial_distance_grid(n, d):
    """generate n-d radial distance grid, assumes input is dimension of signal"""

    l = np.arange(-n//2, n//2)**2
    grid = np.array(np.meshgrid(*[l for _ in range(d)]))
    rdists = np.round(np.sqrt(np.sum(grid, axis=0)))

    return rdists


def sphere_mask(n, d, radius=False):
    """returns sphere mask as boolean"""

    if n % 2 == 0:
        z = np.linspace(-1, 1, n, endpoint=False)
    else:
        z = np.linspace(-(1 - (1/n)), 1 - (1/n), n, endpoint=True) 
    grid = np.array(np.meshgrid(*[z**2 for _ in range(d)]))
    mask = np.sqrt(np.sum(grid, axis=0)) <= 1

    return mask


def signal_convolution(f, g):
    """convolve image f with g, both real LxL arrays"""

    f_hat = np.fft.fftshift(np.fft.fftn(f))
    g_hat = np.fft.fftshift(np.fft.fftn(g))
    z = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(g_hat * f_hat))).real

    return z


def generate_centered_gaussian(N, d=2, sigma=1, normalize=True):
    
    l = np.linspace(-N/2, N/2, N)
    p = np.array(np.meshgrid(*[l for _ in range(d)]))
    g = np.exp(-( np.sum(p**2, axis=0) ) / (2*sigma**2))
    if normalize:
        g = g / np.sum(g)  # normalize to sum = 1
    
    return g


def indicator_3d(N, dtype=np.float64):
    
    # coordinate grid centered at 0
    ax = np.arange(N, dtype=dtype) - (N - 1) / 2
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r2 = x*x + y*y + z*z
    g = 1/(4*np.pi/3)*(r2<= 3*ax[-1]**2)
    
    return g


def gaussian_3d(N, sigma, dtype=np.float64):
    
    # coordinate grid centered at 0
    ax = np.arange(N, dtype=dtype) - (N - 1) / 2
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r2 = x*x + y*y + z*z
    g = np.exp(-0.5 * r2 / (sigma * sigma))
    g = g / np.sum(g) # normalize to sum = 1
    
    return g


def save_pkl(save_path, d):
    with open(save_path, "wb") as file:
        pickle.dump(d, file)


def load_pkl(save_path):
    with open(save_path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data