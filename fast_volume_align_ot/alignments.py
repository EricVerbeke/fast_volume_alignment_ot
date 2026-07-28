import time

import numpy as np

import torch
import torch_harmonics as th

import spherical
import quaternionic

from fast_volume_align_ot import transforms, signals, fle_3d, distances

import scipy.special as spl #for the Euclidean alignment
from scipy.integrate import simpson
from scipy.spatial.transform import Rotation


### TODO:

### change Alignment so that transforms can be done outside of alignment loop 
### (when doing multiple alignments)
### since the transforms class is vectorized
### also then we can define fle only one time

### add option to not save the aligned model (to save memory)
### add option to automatically compute the default grid if not given


class Alignment:

    def __init__(self, f, g, grid, metric, use_correlation=False):

        c, d, R_idx = compute_rotational_distance(f, g, grid, metric)
        
        if use_correlation:
            R = alignment_matrix_from_distances(c, R_idx, use_correlation=True)
        else:
            R = alignment_matrix_from_distances(d, R_idx, use_correlation=False)

        o_est, b_est, a_est = Rotation.from_matrix(R.T).as_euler('zyz')
        g_aligned = transforms.rotate_voxel_zyz_fourier(g, -a_est, -b_est, -o_est)
        
        self.correlations = c
        self.distances = d
        self.R = R
        self.aligned = g_aligned

        
def compute_rotational_distance(f, g, grid, metric):
        
        d = 3
        M = grid.n_phis  
        max_l = M//2

        fle = fle_3d.FLEBasis3D(M, M, 1e-6, max_l=max_l, mode="complex", force_real="True")

        if metric == 'L2':

            ### EV: redefine fle if computing L2
            M = f.shape[0]
            max_l = M//2
            fle = fle_3d.FLEBasis3D(M, M, 1e-6, max_l=max_l, mode="complex", force_real="True")

            c, R_idx, f_norm, g_norm = align_l2(np.flip(f, axis=2), np.flip(g, axis=2), grid, fle)
            d = (f_norm + g_norm - 2*c)/np.sqrt(f_norm*g_norm)

        elif metric == 'SW':

            f_t = transforms.Transform(f, d, grid).inverse_cdf_transform()[0]
            g_t = transforms.Transform(g, d, grid).inverse_cdf_transform()[0]
            
            c, R_idx, f_t_norm, g_t_norm = fast_rotational_correlation(f_t, g_t, grid, fle)
            
            c = c/np.sqrt(f_t_norm*g_t_norm)
            d = (np.sqrt(f_t_norm/g_t_norm) + np.sqrt(g_t_norm/f_t_norm) - 2*c) / (4*np.pi)

        elif metric == 'RFSW':

            f_tp, f_tn = transforms.Transform(f, d, grid, apply_ramp=True).signed_inverse_cdf_transform()
            f_tp, f_tn = f_tp[0], f_tn[0]

            g_tp, g_tn = transforms.Transform(g, d, grid, apply_ramp=True).signed_inverse_cdf_transform()
            g_tp, g_tn = g_tp[0], g_tn[0]

            cp, Rp_idx, f_tp_norm, g_tp_norm = fast_rotational_correlation(f_tp, g_tp, grid, fle)
            cp = cp/np.sqrt(f_tp_norm*g_tp_norm)
            
            cn, Rn_idx, f_tn_norm, g_tn_norm = fast_rotational_correlation(f_tn, g_tn, grid, fle)
            cn = cn/np.sqrt(f_tn_norm*g_tn_norm)

            c = (cp + cn) / 2
            
            d_rfsw_p = (np.sqrt(f_tp_norm/g_tp_norm) + np.sqrt(g_tp_norm/f_tp_norm) - 2*cp) / (4*np.pi)
            d_rfsw_n = (np.sqrt(f_tn_norm/g_tn_norm) + np.sqrt(g_tn_norm/f_tn_norm) - 2*cn) / (4*np.pi)
            
            d = (d_rfsw_p + d_rfsw_n) / 2

            if Rp_idx != Rn_idx:
                print("NOTE: positive and negative distance give different minimum")
            R_idx = Rp_idx
    
        elif metric == 'SC':
            pass    

        return c, d, R_idx        


def alignment_matrix_from_distances(c, R_idx, use_correlation):
    
    if use_correlation:
        xiind,etaind,omegaind = np.unravel_index(np.argmax(c), c.shape)
    else:
        xiind,etaind,omegaind = np.unravel_index(np.argmin(c), c.shape)

    if R_idx == 0:
        R_max = np.eye(3)
    
    else:
        ### xi, eta, omega need to be changed from indices to values
        xi = 2*np.pi*xiind/c.shape[0]
        eta = 2*np.pi*etaind/c.shape[1]
        omega = 2*np.pi*omegaind/c.shape[2]
        
        ### OM: May 28 2026, we can use this if we want, and need a change of the conventions in another part 
        ### the below phi, theta, psi is what the paper says
        ### (see Test_grid_align.ipynb)
        # phi = (xi + np.pi/2)
        # theta = np.pi - eta
        # psi = (omega + np.pi/2)
        
        ### but the below phi, theta, psi are what work.
        ### Not sure why the minus signs are needed...
        phi = -(xi + np.pi/2)
        theta = np.pi - eta
        psi = -(omega + np.pi/2)

        R_max = transforms.rotation_matrix_from_zyz(phi,theta,psi)
    
    return R_max 


def fast_rotational_correlation(f, g, grid, fle):
    """
    f and g are the icdfs of the projection tensors Pf and Pg
    fle is a basis object from FLE3D, needed to call some functions that massage
    the input to the fast spherical harmonics transform. We can get around using this
    by copying some functions from FLE3D, if we want.
    """

    ### EV: need to change this if we want to compute Cramer distance, should use different grid
    
    f = np.squeeze(f.copy())
    g = np.squeeze(g.copy())

    lmax = fle.lmax+1
    
    sht = th.RealSHT(grid.n_thetas, grid.n_phis, lmax=lmax, grid="equiangular", csphase=True).to("cpu")

    ### torch_harmonics wants the order radius, theta, phi
    f_swapped = np.transpose(f,(1,2,0))
    g_swapped = np.transpose(g,(1,2,0))

    fhat = fle.torch_reshape_order_t(sht(torch.DoubleTensor(np.real(f_swapped)).to("cpu")).cpu().numpy())
    ghat = fle.torch_reshape_order_t(sht(torch.DoubleTensor(np.real(g_swapped)).to("cpu")).cpu().numpy())
    fhat = fhat[:,:,:(2*lmax+1)]
    ghat = ghat[:,:,:(2*lmax+1)]
    ### these are now in order radius, l, m, where for fixed radius and l, the m index varies like
    #m=0,-1,1,-2,2,-3,3, ...

    Ilm = np.zeros((lmax,2*lmax+1,2*lmax+1),dtype=np.complex128)
    
    for m1 in range(-lmax, lmax+1):
        m1ind = 2*np.abs(m1) - (m1 < 0)
        for m2 in range(-lmax, lmax+1):
            m2ind = 2*np.abs(m2) - (m2 < 0)

            inds = np.arange(0, grid.n_ts)
            for lind in range(Ilm.shape[0]):

                if grid.t_grid_type == 'uniform':
                    Ilm[lind,m1+lmax,m2+lmax] = simpson(
                        fhat[inds,lind,m1ind]*np.conj(ghat[inds,lind,m2ind]), x=grid.ts[inds]
                    )

                if grid.t_grid_type == 'gauss-legendre':
                    Ilm[lind,m1+lmax,m2+lmax] = np.sum(
                        fhat[inds,lind,m1ind]*np.conj(ghat[inds,lind,m2ind])*grid.wg[inds]
                    )

    f_t_norm = 0
    g_t_norm = 0
    
    for m in range(-lmax, lmax+1):
        mind = 2*np.abs(m) - (m < 0)
        inds = np.arange(0, grid.n_ts)
        for lind in range(Ilm.shape[0]):
            if grid.t_grid_type == 'uniform':
                f_t_norm = f_t_norm + simpson(
                    fhat[inds,lind,mind]*np.conj(fhat[inds,lind,mind]), x=grid.ts[inds]
                ).real
                g_t_norm = g_t_norm + simpson(
                    ghat[inds,lind,mind]*np.conj(ghat[inds,lind,mind]), x=grid.ts[inds]
                ).real

            if grid.t_grid_type == 'gauss-legendre':
                f_t_norm = f_t_norm + np.sum(
                    fhat[inds,lind,mind]*np.conj(fhat[inds,lind,mind])*grid.wg[inds]
                ).real

                g_t_norm = g_t_norm + np.sum(
                    ghat[inds,lind,mind]*np.conj(ghat[inds,lind,mind])*grid.wg[inds]
                ).real
                    
    wigner = spherical.Wigner(lmax)
    R = quaternionic.array.from_euler_angles(0, np.pi/2, 0)
    d = wigner.D(R)

    dlm = np.zeros((lmax, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    for m1 in range(-lmax, lmax+1):
        for m2 in range(-lmax, lmax+1):
            for l in range(lmax):
                if abs(m1) <= l and abs(m2) <= l:
                    dlm[l,m1+lmax,m2+lmax] = d[wigner.Dindex(l, m2, m1)]

    T = np.zeros((2*lmax+1, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    for m in range(-lmax, lmax+1):
        for h in range(-lmax, lmax+1):
            for mp in range(-lmax, lmax+1):
                T[m+lmax,h+lmax,mp+lmax] = np.sum(dlm[:,m+lmax,h+lmax]*dlm[:,h+lmax,mp+lmax]*Ilm[:,m+lmax,mp+lmax])

    ### compute the correlation with no rotation    
    tmpcorr = 0
    for l in range(Ilm.shape[0]):
        tmpcorr += np.trace(Ilm[l,:,:]).real

    ### need to multiply by T.size, because the ifft divides by this number
    c = T.size*np.fft.ifftn(np.fft.ifftshift(T)).real

    # print('correlation R=0: ', np.amax(tmpcorr))
    # print('all correlation: ', np.amax(c))

    if np.amax(tmpcorr) >= np.amax(c):
        c = np.zeros(T.shape)   # EV: actually doesn't need to be same dimension
        c[0,0,0] = tmpcorr
        R_idx = 0
    
    else:
        R_idx = 1

    return c, R_idx, f_t_norm, g_t_norm


def align_l2(f, g, grid, fle):
    
    N = f.shape[0]
    h = 1/np.floor((N+1+N%2)/2)
    cf = fle.evaluate_t(f) * fle.h
    cg = fle.evaluate_t(g) * fle.h

    n_points = f.shape[0]
    lmax = fle.lmax+1

    z = grid.ts

    ### STEP 1: build a coefficient tensor with indices k,l,m,
    ### where the m index varies like m=0,-1,1,-2,2,-3,3, ...
    cf2 = np.zeros((fle.kmax,lmax,2*lmax+1),dtype=np.complex128)
    cg2 = np.zeros((fle.kmax,lmax,2*lmax+1),dtype=np.complex128)
    lmds = np.zeros((fle.kmax,lmax),dtype=np.complex128)
    cs = np.zeros((fle.kmax,lmax),dtype=np.complex128)
    for i in range(len(cf)):
        k = fle.ks[i]
        l = fle.ls[i]
        m = fle.ms[i]
        md = 2*np.abs(m) - (m < 0)
        cf2[k-1,l,md] = cf[i]
        cg2[k-1,l,md] = cg[i]
        lmds[k-1,l] = fle.lmds[i]
        cs[k-1,l] = fle.cs[i]
            
    ### STEP 2: sum over the bessel functions to get radius,l,m, 
    ### where the m index varies like m=0,-1,1,-2,2,-3,3, ...
    fhat = np.zeros((n_points,lmax,2*lmax+1),dtype=np.complex128)
    ghat = np.zeros((n_points,lmax,2*lmax+1),dtype=np.complex128)
    
    for l in range(cf2.shape[1]):
        tmpf = np.zeros((fhat.shape[0],fhat.shape[2]), dtype=np.complex128)
        tmpg = np.zeros((fhat.shape[0],fhat.shape[2]), dtype=np.complex128)
        for k in range(cf2.shape[0]):
            ### OM: this can be replaced by a fast Bessel 
            lmd = lmds[k,l]
            c = cs[k,l]
            sb = c*spl.spherical_jn(l, lmd * np.abs(z))
            sb[-1] = 0
            
            ### this is always true at z = 1,
            ### but scipy sometimes makes sb nan here,
            ### so let's fix it manually
            # sb[0:len(sb)//2] = (-1)**l*sb[0:len(sb)//2]

            tmpf += (sb.reshape(-1,1))@(cf2[k,l,:].reshape(1,-1))
            tmpg += (sb.reshape(-1,1))@(cg2[k,l,:].reshape(1,-1))
        fhat[:,l,:] = tmpf
        ghat[:,l,:] = tmpg

    #STEP 3: run the below.
    #### THE BELOW IS COPIED FROM fast_rotational_correlation.
    #### PROBABLY BETTER TO SEPARATE IT OUT AS A SEPARATE FUNCTION.
    Ilm = np.zeros((lmax,2*lmax+1,2*lmax+1),dtype=np.complex128)
    for m1 in range(-lmax, lmax+1):
        m1ind = 2*np.abs(m1) - (m1 < 0)
        for m2 in range(-lmax, lmax+1):
            m2ind = 2*np.abs(m2) - (m2 < 0)
            inds = np.arange(0,len(z))
            for lind in range(Ilm.shape[0]):
                Ilm[lind,m1+lmax,m2+lmax] = simpson(
                    fhat[inds,lind,m1ind]*np.conj(ghat[inds,lind,m2ind])*np.abs(z[inds])**2, x = z[inds]
                )

                # if grid.t_grid_type == "gauss-legendre":
                #     Ilm[lind,m1+lmax,m2+lmax] = np.sum(
                #         fhat[inds,lind,m1ind]*np.conj(ghat[inds,lind,m2ind])*np.abs(z[inds])**2*grid.wg[inds]
                #     )
                
    f_norm = 0
    g_norm = 0
    
    for m in range(-lmax, lmax+1):
        mind = 2*np.abs(m) - (m < 0)
        inds = np.arange(0,len(z))
        for lind in range(Ilm.shape[0]):
            f_norm = f_norm + simpson(
                fhat[inds,lind,mind]*np.conj(fhat[inds,lind,mind])*np.abs(z[inds])**2, x = z[inds]
            ).real
            g_norm = g_norm + simpson(
                ghat[inds,lind,mind]*np.conj(ghat[inds,lind,mind])*np.abs(z[inds])**2, x = z[inds]
            ).real

            if grid.t_grid_type == 'uniform':
                f_norm = f_norm + simpson(
                    fhat[inds,lind,mind]*np.conj(fhat[inds,lind,mind])*np.abs(z[inds])**2, x = z[inds]
                ).real
                g_norm = g_norm + simpson(
                    ghat[inds,lind,mind]*np.conj(ghat[inds,lind,mind])*np.abs(z[inds])**2, x = z[inds]
                ).real

            if grid.t_grid_type == 'gauss-legendre':
                f_norm = f_norm + np.sum(
                    fhat[inds,lind,mind]*np.conj(fhat[inds,lind,mind])*grid.wg[inds]*np.abs(z[inds])**2
                ).real

                g_norm = g_norm + np.sum(
                    ghat[inds,lind,mind]*np.conj(ghat[inds,lind,mind])*grid.wg[inds]*np.abs(z[inds])**2
                ).real

    wigner = spherical.Wigner(lmax)
    R = quaternionic.array.from_euler_angles(0, np.pi/2, 0)
    d = wigner.D(R)

    dlm = np.zeros((lmax, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    for m1 in range(-lmax, lmax+1):
        for m2 in range(-lmax, lmax+1):
            for l in range(lmax):
                if abs(m1) <= l and abs(m2) <= l:
                    dlm[l,m1+lmax,m2+lmax] = d[wigner.Dindex(l, m1, m2)]

    T = np.zeros((2*lmax+1, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    for m in range(-lmax, lmax+1):
        for h in range(-lmax, lmax+1):
            for mp in range(-lmax, lmax+1):
                T[m+lmax,h+lmax,mp+lmax] = np.sum(dlm[:,m+lmax,h+lmax]*dlm[:,h+lmax,mp+lmax]*Ilm[:,m+lmax,mp+lmax])
                
    ### compute the correlation with no rotation    
    tmpcorr = 0
    for l in range(Ilm.shape[0]):
        tmpcorr += np.trace(Ilm[l,:,:]).real

    ### need to multiply by T.size, because the ifft divides by this number
    c = T.size*np.fft.ifftn(np.fft.ifftshift(T)).real

    # print('correlation R=0: ', np.amax(tmpcorr))
    # print('all correlation: ', np.amax(c))

    if np.amax(tmpcorr) >= np.amax(c):
        c = np.zeros(T.shape)   # EV: actually doesn't need to be same dimension
        c[0,0,0] = tmpcorr
        R_idx = 0
    
    else:
        R_idx = 1

    return c, R_idx, f_norm, g_norm