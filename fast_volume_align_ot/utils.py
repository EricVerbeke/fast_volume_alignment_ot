import numpy as np

from scipy import ndimage as ndi


def zero_pad_voxel_array_to_size(f, L):
    pass


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

    rdists = radial_distance_grid(n, d)
    
    if not radius:
        center = [n//2 for n in rdists.shape]
        radius = np.amin(center)
        
    mask = rdists <= radius
    
    return mask


def signal_convolution(f, g):
    """convolve image f with g, both real LxL arrays"""

    ### note: doesn't account for circular convolution (wrap around)
    ### NEED TO UPDATE THIS FUNCTION
    ### do I need a conjugation here somewhere?

    f_hat = np.fft.fftshift(np.fft.fftn(f))
    g_hat = np.fft.fftshift(np.fft.fftn(g))
    z = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(g_hat * f_hat))).real

    return z


def generate_centered_gaussian(L=128, d=2, sigma=1):

    ### add option to normalize (to 1, by L or none)
    
    l = np.linspace(-L/2, L/2, L)
    p = np.array(np.meshgrid(*[l for _ in range(d)]))
    g = np.exp(-( np.sum(p**2, axis=0) ) / (2*sigma**2))
    g = g / np.sum(g)  # normalize to sum = 1
    
    return g


def generate_3d_triangle_gaussian_shape(L, sigma, rescale=False):
    """
    input: L = integer size of grid
    notes: L needs to be in {6n + 1}, e.g., 31,  37,  43,  49,  55,  61,  67,  73, 79,  85
    output: a 3D voxel grid with 3 Gaussians in following shape
    (x-projection to zy-plane)
     -------------
    |             |
    |   x     x   |
    |             |
    |      x      |
    |             |
     -------------
    """

    f = np.zeros((L,L,L))
    
    x_1 = L//2
    x_2 = L//3
    f[x_1, x_2, x_2] = 1
    f[x_1, x_2, 2*x_2] = 1
    f[x_1, 2*x_2, x_1] = 1
    
    g = generate_centered_gaussian(L, d=3, sigma=sigma)
    f_conv = signal_convolution(f, g)
    f_conv = f_conv / np.sum(f_conv)  # normalize to sum = 1

    if rescale:
        f_conv = L * f_conv

    return f_conv