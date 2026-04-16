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


def generate_centered_gaussian(N, d=2, sigma=1):

    ### add option to normalize (to 1, by L or none)
    
    l = np.linspace(-N/2, N/2, N)
    p = np.array(np.meshgrid(*[l for _ in range(d)]))
    g = np.exp(-( np.sum(p**2, axis=0) ) / (2*sigma**2))
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
    

def generate_3d_triangle_gaussian_shape(N, sigma, rescale=False):
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

    f = np.zeros((N,N,N))
    
    x_1 = N//2
    x_2 = N//3
    f[x_1, x_2, x_2] = 1
    f[x_1, x_2, 2*x_2] = 1
    f[x_1, 2*x_2, x_1] = 1
    
    g = generate_centered_gaussian(N, d=3, sigma=sigma)
    f_conv = signal_convolution(f, g)
    f_conv = f_conv / np.sum(f_conv)  # normalize to sum = 1

    if rescale:
        f_conv = N * f_conv

    return f_conv


def sequential_array(d, n):

    return np.arange(n**d).reshape(*[n for _ in range(d)])


def cartesian_to_voxel(p, N):
    """
    points (p): (n_p, 3) array of (x,y,z) in [-1,1]
    N: voxel size (assume equal dimensions and spacing) nx = ny = nz = N
        axis 0 = x  (out of screen)
        axis 1 = z  (up) but flipped so z=+1 maps to index 0
        axis 2 = y  (right)
    Returns: (n_p, 3) integer indices (ix, iz, iy)
    Notes: int is equivalent to 'floor' nearest neighbor interpolation (I think?)
    """

    x, y, z = p[..., 0], p[..., 1], p[..., 2]

    # x flip: x = +1 -> ix = 0, x = -1 -> ix = nx-1
    ix = ((1 - x) * 0.5 * (N - 1)).astype(int)

    # z flip: z = +1 -> iz = 0, z = -1 -> iz = nz-1
    iz = ((1 - z) * 0.5 * (N - 1)).astype(int)

    # y normal: y = -1 -> 0, y = +1 -> ny-1
    iy = ((y + 1) * 0.5 * (N - 1)).astype(int)

    return np.stack([ix, iy, iz])


def place_points_into_volume(idx, N, value=1):
    """
    Write voxels into a NumPy volume with geometry-preserving layout:
        volume[z, y, x]
    idx: (3, n_points) voxel indices
    """
    volume = np.zeros((N, N, N), dtype=np.float32)

    ix, iy, iz = idx  # unpack integer indices

    # Geometry-preserving placement: z->axis0, y->axis1, x->axis2
    volume[ix, iz, iy] = value
    
    return volume


def generate_voxel_array_from_points(p, N, value=1):

    idx = cartesian_to_voxel(p, N)
    vol = place_points_into_volume(idx, N, value)

    return vol


# def points_to_voxels(p, N):
#     """
#     converts points (x,y,z) in [-1,1]^3 to voxel indices in [0, N-1]^3
#     input: p is 3 X n_points array
#     """

#     ### this is probably wrong too
#     p_n = (p + 1) / 2 * (N-1)
#     v_idx = np.round(p_n).astype('int')  # need to use floor? centering for even?

#     return v_idx


# def fill_voxel_array(v_idx, N, value=1):
#     """fill voxel array at specified index with value"""

#     f = np.zeros((N,N,N))

#     vx = v_idx[0, :]
#     vy = v_idx[1, :]
#     vz = v_idx[2, :]

#     ### how to choose conventions so that Cartesian and numpy grid match ?
#     # f[-vz, -vy, vx] = value
#     f[-vx, -vz, vy] = value
    
#     ### probably negative is wrong , flipping axes should be N - 1 - vn?
#     # vx_np = N - 1 - vx
#     # vz_np = N - 1 - vz
#     # f[vx_np, vz_np, vy] = value
    
#     return f


# def generate_voxel_array_from_points(p, N, value=1):
    
#     v_idx = points_to_voxels(p, N)
#     f = fill_voxel_array(v_idx, N, value)

#     return f