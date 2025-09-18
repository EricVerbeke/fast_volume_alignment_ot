import numpy as np


def return_min_angles(dists, thetas, phis):

    min_theta_idx, min_phi_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    
    min_theta = thetas[min_theta_idx]
    min_phi = phis[min_phi_idx]

    return min_theta, min_phi