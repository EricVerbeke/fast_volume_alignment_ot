import time

import numpy as np

import torch
import torch_harmonics as th

import spherical
import quaternionic

from fast_volume_align_ot import transforms, signals, fle_3d

import scipy.special as spl #for the Euclidean alignment
from scipy.integrate import simpson
from scipy.spatial.transform import Rotation


### EV: implement this differently later so that the grid doesn't need to be redefined
###     but also cost is probably negligable


### EV: change Alignment so that transforms can be vectorized and done outside of loop (when doing multiple alignments)


class Alignment:

    def __init__(self, f, g, metric, use_correlation=False):

        c, d, R_idx = compute_rotational_distance(f, g, metric)
        
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

        
def compute_rotational_distance(f, g, metric):

        N = f.shape[0]
        M = f.shape[0]
        max_l = M//2

        fle = fle_3d.FLEBasis3D(M, M, 1e-6, max_l=max_l, mode="complex", force_real="True")

        ### EV: these are recomputed, combine with previous functions/classes
        t = np.linspace(0, 1, N, endpoint=True) 
        phis = np.linspace(0, 2*np.pi, N, endpoint=False)
        thetas = np.linspace(0, np.pi, N, endpoint=True)

        sinthetas = np.sin(thetas)[np.newaxis, np.newaxis, :]  ### EV: DOUBLE CHECK THIS!!!

        dVsphere = (t[1]-t[0])*(thetas[1]-thetas[0])*(phis[1]-phis[0]) 

        ### for Euclidean distance
        h = 1/np.floor( (N+1+N%2)/2 )
        dV = h**3

        if metric == 'L2':
            
            # cf = fle.evaluate_t(np.flip(f, axis=2)) * fle.h
            # cg = fle.evaluate_t(np.flip(g, axis=2)) * fle.h

            f_norm = np.sqrt(np.sum(f**2))
            g_norm = np.sqrt(np.sum(g**2))  
                    
            c, R_idx = align_l2(np.flip(f, axis=2), np.flip(g, axis=2), fle)
            d = (f_norm**2 + g_norm**2) * dV - 2*c

        elif metric == 'SW':

            f_t = transforms.Transform(f, d=3).inverse_cdf_transform()[0]
            g_t = transforms.Transform(g, d=3).inverse_cdf_transform()[0]
            
            f_t_norm = np.sqrt(np.sum(f_t**2 * sinthetas))
            g_t_norm = np.sqrt(np.sum(g_t**2 * sinthetas))
            
            c, R_idx = fast_rotational_correlation(f_t, g_t, fle)
            d = ((f_t_norm**2 + g_t_norm**2) * dVsphere - 2*c) / (4*np.pi)

        elif metric == 'RFSW':

            f_tp, f_tn = transforms.Transform(f, d=3, apply_ramp=True).signed_inverse_cdf_transform()
            f_tp_norm = np.sqrt(np.sum(f_tp[0]**2 * sinthetas))
            f_tn_norm = np.sqrt(np.sum(f_tn[0]**2 * sinthetas))
            
            g_tp, g_tn = transforms.Transform(g, d=3, apply_ramp=True).signed_inverse_cdf_transform()
            g_tp_norm = np.sqrt(np.sum(g_tp[0]**2 * sinthetas))
            g_tn_norm = np.sqrt(np.sum(g_tn[0]**2 * sinthetas))

            cp, Rp_idx = fast_rotational_correlation(f_tp[0], g_tp[0], fle)
            cn, Rn_idx = fast_rotational_correlation(f_tn[0], g_tn[0], fle)
            c = (cp + cn) / 2
            d_rfsw_p = ((f_tp_norm**2 + g_tp_norm**2) * dVsphere - 2*cp) / (4*np.pi)
            d_rfsw_n = ((f_tn_norm**2 + g_tn_norm**2) * dVsphere - 2*cn) / (4*np.pi)
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
        #xi, eta, omega need to be changed from indices to values
        xi = 2*np.pi*xiind/c.shape[0]
        eta = 2*np.pi*etaind/c.shape[1]
        omega = 2*np.pi*omegaind/c.shape[2]
    
        # # the below phi, theta, psi is what the paper says
        # phi = xi + np.pi/2
        # theta = np.pi - eta
        # psi = omega + np.pi/2
        
        # but the below phi, theta, psi are what work.
        # Not sure why the minus signs are needed...
        phi = -(xi + np.pi/2)
        theta = np.pi - eta
        psi = -(omega + np.pi/2)

        # print("checking angles",phi, theta, psi)
        R_max = transforms.rotation_matrix_from_zyz(phi,theta,psi)
    
    return R_max 


def fast_rotational_correlation(f, g, fle):
    #icf and icg are the icdfs of the projection tensors Pf and Pg
    #fle is a basis object from FLE3D, needed to call some functions that massage
    #the input to the fast spherical harmonics transform. We can get around using this
    #by copying some functions from FLE3D, if we want.
    
    #Step 1: replace icf and icg by 1/abs(z)*icf and 1/abs(z)*icg
    #maybe we could make this z into a self.z in the __init__?

    ### EV: not sure why we need the copy but keeping it for now
    
    f = np.squeeze(f.copy())
    g = np.squeeze(g.copy())

    n_phi, n_points, n_theta = f.shape
    
    lmax = fle.lmax+1
    
    ### EV: this gets recomputed but is minimal cost, can combine in one object later
    # if n_points % 2 == 0:
    #     z = np.linspace(-1, 1, n_points, endpoint=False)
    # else:
    #     z = np.linspace(-(1 - (1/n_points)), 1 - (1/n_points), n_points, endpoint=True) 

    z = np.linspace(0, 1, n_points, endpoint=True)
    
    for i in range(f.shape[1]):
        #the entry for z = 0 is not rotated by the alignment so its value does not matter
        #(unless we use a spherical harmonics transform that does some interpolation)
        #this 1/np.abs(z[i]) is actually cancelled by something below, so we should maybe remove it
        #here and below.
        if z[i] != 0:
            f[:,i,:] = f[:,i,:] / (np.abs(z[i]))
            g[:,i,:] = g[:,i,:] / (np.abs(z[i]))
        
    #Step 2: compute spherical harmonics coefficients
    sht = th.RealSHT(n_theta, n_phi, lmax=lmax, grid="equiangular", csphase=True).to("cpu")

    #torch_harmonics wants the order radius, theta, phi
    f_swapped = np.transpose(f,(1,2,0))
    g_swapped = np.transpose(g,(1,2,0))

    #Question: icf and icg are real-valued, right? Torch-harmonics only likes real inputs
    #If not, we just have to call it twice, using the imaginary parts for the second call.
    ### EV: yes, real-valued. but we might want option for imaginary to compute Euclidean dist from the FT
    fhat = fle.torch_reshape_order_t(sht(torch.DoubleTensor(np.real(f_swapped)).to("cpu")).cpu().numpy())
    ghat = fle.torch_reshape_order_t(sht(torch.DoubleTensor(np.real(g_swapped)).to("cpu")).cpu().numpy())
    fhat = fhat[:,:,:(2*lmax+1)]
    ghat = ghat[:,:,:(2*lmax+1)]
    #these are now in order radius, l, m, where for fixed radius and l, the m index varies like
    #m=0,-1,1,-2,2,-3,3, ...

    ### EV: keeping this here for now but can delete later?
    # import matplotlib.pyplot as plt
    # plt.imshow(np.abs(fhat[10,:,:]))
    # plt.colorbar()
    # print(fhat[32,1:5,1:5])
    # print(np.linalg.norm(fhat))
    # plt.show()
   

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(f[0,:,0]*2*np.sqrt(np.pi))
    # plt.plot(fhat[:,0,0])
    # plt.figure()
    # plt.plot(f[0,:,0]*2*np.sqrt(np.pi))
    # plt.figure()
    # plt.plot(fhat[:,0,0])
    # plt.figure()
    # plt.plot(f[0,:,0]*2*np.sqrt(np.pi)-fhat[:,0,0])
    #these agree fairly well, but why aren't they symmetric around the midpoint?
    #because f_gt is not necessarily radially symmetric... but it should be for gaussian f.
    #no, it should not because of the icdf
    
    #Step 3: compute the integral
    #the d^\ell_{mh} are non-zero only if |m| <= \ell, |h| <= \elll
    #so we can zero-pad them and make tensors of size lmax*(2*lmax+1)*(2*lmax+1)
    #and then just take the sum over the first dimension of d^\ell_{mh} * d^\ell_{hm'}*I^\ell_{mm'}
    #and store this in T(m,h,m')

    Ilm = np.zeros((lmax,2*lmax+1,2*lmax+1),dtype=np.complex128)
    #we need to remember to multiply by r^2, which cancels out the 1/abs(z) above...
    for m1 in range(-lmax, lmax+1):
        m1ind = 2*np.abs(m1) - (m1 < 0)
        for m2 in range(-lmax, lmax+1):
            m2ind = 2*np.abs(m2) - (m2 < 0)
            # inds = np.arange(n_points//2-1,len(z))
            inds = np.arange(1,len(z))
            for lind in range(Ilm.shape[0]):
                Ilm[lind,m1+lmax,m2+lmax] = simpson(
                    fhat[inds,lind,m1ind]*np.conj(ghat[inds,lind,m2ind])*np.abs(z[inds])**2, x = z[inds]
                )
            #wfhat, ghat have a different ordering. m1ind and m2ind fix this
            # for i in range(fhat.shape[0]):
            #     #should this actually only be for positive z? also need to divide by delta z...
            #     #we can use better quadrature here to get increased accuracy
            #     # Ilm[:,m1+lmax,m2+lmax] += fhat[i,:,m1ind]*np.conj(ghat[i,:,m2ind])*np.abs(z[i])**2*(z[1]-z[0])/2
            #     if z[i]>z[n_points//2-1]:#z[i] > 0:
            #         Ilm[:,m1+lmax,m2+lmax] += fhat[i,:,m1ind]*np.conj(ghat[i,:,m2ind])*np.abs(z[i])**2*(z[1]-z[0])
            #         #I think there is a mismatch in the quadrature here

    wigner = spherical.Wigner(lmax)
    R = quaternionic.array.from_euler_angles(0, np.pi/2, 0)
    d = wigner.D(R)

    dlm = np.zeros((lmax, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    for m1 in range(-lmax, lmax+1):
        for m2 in range(-lmax, lmax+1):
            for l in range(lmax):
                if abs(m1) <= l and abs(m2) <= l:
                    # dlm[l,m1+lmax,m2+lmax] = d[wigner.Dindex(l, m1, m2)]
                    dlm[l,m1+lmax,m2+lmax] = d[wigner.Dindex(l, m2, m1)]
            # dlm[:,m1+lmax,m2+lmax] = wigner_dl(0, lmax, m1, m2, np.pi/2)
            #computes all wigner d-matrices for l between 0 and lmax at fixed m1,m2 and angle pi/2

    T = np.zeros((2*lmax+1, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    for m in range(-lmax, lmax+1):
        for h in range(-lmax, lmax+1):
            for mp in range(-lmax, lmax+1):
                T[m+lmax,h+lmax,mp+lmax] = np.sum(dlm[:,m+lmax,h+lmax]*dlm[:,h+lmax,mp+lmax]*Ilm[:,m+lmax,mp+lmax])


    ### EV: computes the correlation with no rotation
    ### EV: simple fix just return special value, pass to alignment_matrix, return R = I
    
    tmpcorr = 0
    for l in range(Ilm.shape[0]):
        tmpcorr += np.trace(Ilm[l,:,:]).real
    

    #Step 4: do an IFFT
    ###### c = np.fft.fftshift(sifftn(np.fft.ifftshift(T), workers=-1))
    #TODO: WHICH ONE OF THESE IS FASTER??
    c = T.size*np.fft.ifftn(np.fft.ifftshift(T)).real
    #need to multiply by T.size, because the ifft divides by this number
    # c = sifftn(np.fft.ifftshift(T), workers=-1)
    # c = sifftn(np.fft.ifftshift(T), workers=1)

    print('correlation R=0: ', np.amax(tmpcorr))
    print('all correlation: ', np.amax(c))

    if np.amax(tmpcorr) >= np.amax(c):
        c = np.zeros(T.shape)   # EV: actually doesn't need to be same dimension
        c[0,0,0] = tmpcorr
        R_idx = 0
    
    else:
        R_idx = 1
    
    # print("everything else: ", time.time() - t)

    return c, R_idx


def align_l2(f, g, fle):
    
    # t = time.time()
    
    N = f.shape[0]
    h = 1/np.floor( (N+1+N%2)/2)
    cf = fle.evaluate_t(f) * fle.h
    cg = fle.evaluate_t(g) * fle.h

    n_points = f.shape[0]
    lmax = fle.lmax+1

    # ### EV: this gets recomputed but is minimal cost, can combine in one object later
    if n_points % 2 == 0:
        z = np.linspace(-1, 1, n_points, endpoint=False)
    else:
        z = np.linspace(-(1 - (1/n_points)), 1 - (1/n_points), n_points, endpoint=True)


    #STEP 1: build a coefficient tensor with indices k,l,m,
    #where the m index varies like m=0,-1,1,-2,2,-3,3, ...
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


    # print("evaluate_t, step 1: ", time.time() - t)
    
    #STEP 2: sum over the bessel functions to get radius,l,m, 
    #where the m index varies like m=0,-1,1,-2,2,-3,3, ...

    # t = time.time()
    
    fhat = np.zeros((n_points,lmax,2*lmax+1),dtype=np.complex128)
    ghat = np.zeros((n_points,lmax,2*lmax+1),dtype=np.complex128)

    for l in range(cf2.shape[1]):
        for md in range(cf2.shape[2]):
            # m = ((md+1)//2)*(-1)**md
            tmpf = np.zeros(fhat.shape[0], dtype=np.complex128)
            tmpg = np.zeros(ghat.shape[0], dtype=np.complex128)
            for k in range(cf2.shape[0]):
                #this can be replaced by a fast Bessel thing
                lmd = lmds[k,l]
                c = cs[k,l]
                sb = c*spl.spherical_jn(l, lmd * np.abs(z))
                
                sb[0:len(sb)//2] = (-1)**l*sb[0:len(sb)//2]
                tmpf += cf2[k,l,md]*sb
                tmpg += cg2[k,l,md]*sb

            fhat[:,l,md] = tmpf
            ghat[:,l,md] = tmpg
            
    # print("step 2: ", time.time() - t)


    #STEP 3: run the below.
    #### THE BELOW IS COPIED FROM fast_rotational_correlation.
    #### PROBABLY BETTER TO SEPARATE IT OUT AS A SEPARATE FUNCTION.
    Ilm = np.zeros((lmax,2*lmax+1,2*lmax+1),dtype=np.complex128)
    for m1 in range(-lmax, lmax+1):
        m1ind = 2*np.abs(m1) - (m1 < 0)
        for m2 in range(-lmax, lmax+1):
            m2ind = 2*np.abs(m2) - (m2 < 0)
            #wfhat, ghat have a different ordering. m1ind and m2ind fix this
            for i in range(fhat.shape[0]):
                #we can use better quadrature here to get increased accuracy
                if z[i] > 0:
                    Ilm[:,m1+lmax,m2+lmax] += fhat[i,:,m1ind]*np.conj(ghat[i,:,m2ind])*np.abs(z[i])**2*(z[1]-z[0])
                    #maybe improve the quadrature...

    ## Compute the correlation analytically, using the formula in the fast rotational matching paper
    
    wigner = spherical.Wigner(lmax)
    R = quaternionic.array.from_euler_angles(0, np.pi/2, 0)
    d = wigner.D(R)

    dlm = np.zeros((lmax, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    for m1 in range(-lmax, lmax+1):
        for m2 in range(-lmax, lmax+1):
            for l in range(lmax):
                if abs(m1) <= l and abs(m2) <= l:
                    dlm[l,m1+lmax,m2+lmax] = d[wigner.Dindex(l, m1, m2)]
                    # dlm[l,m1+lmax,m2+lmax] = d[wigner.Dindex(l, m2, m1)]
            # dlm[:,m1+lmax,m2+lmax] = wigner_dl(0, lmax, m1, m2, np.pi/2)
            #computes all wigner d-matrices for l between 0 and lmax at fixed m1,m2 and angle pi/2

    T = np.zeros((2*lmax+1, 2*lmax+1, 2*lmax+1),dtype=np.complex128)
    for m in range(-lmax, lmax+1):
        for h in range(-lmax, lmax+1):
            for mp in range(-lmax, lmax+1):
                T[m+lmax,h+lmax,mp+lmax] = np.sum(dlm[:,m+lmax,h+lmax]*dlm[:,h+lmax,mp+lmax]*Ilm[:,m+lmax,mp+lmax])
                
    
    ### EV: computes the correlation with no rotation
    ### EV: simple fix just return special value, pass to alignment_matrix, return R = I
    
    tmpcorr = 0
    for l in range(Ilm.shape[0]):
        tmpcorr += np.trace(Ilm[l,:,:]).real


    #Step 4: do an IFFT
    ###### c = np.fft.fftshift(sifftn(np.fft.ifftshift(T), workers=-1))
    #TODO: WHICH ONE OF THESE IS FASTER??
    c = T.size*np.fft.ifftn(np.fft.ifftshift(T)).real
    #need to multiply by T.size, because the ifft divides by this number
    # c = sifftn(np.fft.ifftshift(T), workers=-1)
    # c = sifftn(np.fft.ifftshift(T), workers=1)

    print('correlation R=0: ', np.amax(tmpcorr))
    print('all correlation: ', np.amax(c))

    if np.amax(tmpcorr) >= np.amax(c):
        c = np.zeros(T.shape)   # EV: actually doesn't need to be same dimension
        c[0,0,0] = tmpcorr
        R_idx = 0
    
    else:
        R_idx = 1
    
    # print("everything else: ", time.time() - t)

    return c, R_idx

