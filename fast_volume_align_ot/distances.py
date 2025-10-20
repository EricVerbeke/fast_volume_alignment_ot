import numpy as np

### my module
from fast_volume_align_ot import utils


### TODO:
###  - need to check normalizing constants on distances
###  - need to add volume elements on 3D distance


def sliced_distance_3d(U, V):

    return np.mean((U - V)**2)


def signed_sliced_distance_3d(Up, Vp, Un, Vn):

    d_p = sliced_distance_3d(Up, Vp)
    d_n = sliced_distance_3d(Un, Vn)

    d = (d_p + d_n) / 2

    return d


def fast_rotational_distance_2d(U, V):
    
    U_norm = np.linalg.norm(U)**2
    V_norm = np.linalg.norm(V)**2

    Uj_hat = np.fft.fft(U, axis=1)
    Vj_hat = np.conj(np.fft.fft(V, axis=1))

    UV_l = np.fft.ifft(Uj_hat * Vj_hat, axis=1).real
    UV_l_sum = np.sum(UV_l, axis=0)

    dists = U_norm + V_norm - 2*UV_l_sum
    
    return dists


def fast_signed_rotational_distances_2d(Up, Vp, Un, Vn):

    d_p = fast_rotational_distance_2d(Up, Vp)
    d_n = fast_rotational_distance_2d(Un, Vn)

    d = (d_p + d_n) / 2

    return d


def rotational_euclidean_distance_2d(f, g, angles):
 
    return np.array([np.linalg.norm(f - utils.rotate_image(g, -a))**2 for a in angles])


def return_min_angles(dists, thetas, phis):

    min_theta_idx, min_phi_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    
    min_theta = thetas[min_theta_idx]
    min_phi = phis[min_phi_idx]

    return min_theta, min_phi