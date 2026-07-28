import numpy as np

from scipy.integrate import simpson


def compute_3d_density_volume_norm(f, grid):
    ### EV: returns the squared norm
    ### EV: assumes equal spaced grids for the input volume

    tmp1 = simpson(f**2, x=grid.zs, axis=2)   
    tmp2 = simpson(tmp1, x=grid.zs, axis=0)
    f_norm = simpson(tmp2, x=grid.zs, axis=0)

    return f_norm


def compute_3d_density_l2_distance(f, g, grid):
    ### EV: returns the squared norm
    ### EV: assumes equal spaced grids for the input volume

    tmp1 = simpson((f-g)**2, x=grid.zs, axis=2)   
    tmp2 = simpson(tmp1, x=grid.zs, axis=0)
    f_norm = simpson(tmp2, x=grid.zs, axis=0)

    return f_norm


def compute_3d_icdf_tensor_norm(f_t, grid):
    ### EV: returns the squared norm

    ### OM: these seem to be what we need
    thetas = np.linspace(0, np.pi, len(grid.thetas), endpoint=True)
    phis = np.linspace(0, 2*np.pi, len(grid.phis), endpoint=True)
    tmp1 = simpson(f_t**2 * np.sin(thetas[np.newaxis, np.newaxis, :]), x=grid.thetas, axis=2)
    if grid.t_grid_type == 'uniform':
        tmp2 = simpson(tmp1, x=grid.ts, axis=1) ### OM: these should all be 1, no? the order is phis, radius, theta
    elif grid.t_grid_type == 'gauss-legendre':
        tmp2 = np.sum(tmp1*(grid.wg.reshape(1,-1)),axis=1)
        
    f_t_norm = simpson(tmp2, x=phis, axis=0)

    return f_t_norm


def compute_3d_icdf_inner_product(f_t, g_t, grid):
    
    tmp1 = simpson((f_t * g_t) * np.sin(grid.thetas[np.newaxis, np.newaxis, :]), x=grid.thetas, axis=2)   
    tmp2 = simpson(tmp1, x=grid.ts, axis=0)
    sw_corr = simpson(tmp2, x=grid.phis, axis=0)

    return sw_corr


def compute_3d_sliced_wasserstein_distance(f_t, g_t, grid):
    ### EV: returns the squred distance

    tmp1 = simpson((f_t - g_t)**2 * np.sin(grid.thetas[np.newaxis, np.newaxis, :]), x=grid.thetas, axis=2)   
    tmp2 = simpson(tmp1, x=grid.ts, axis=0)
    sw_dist = simpson(tmp2, x=grid.phis, axis=0)

    return sw_dist    