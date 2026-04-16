import numpy as np

import finufft

import torch
import torch_harmonics as th
import time
import scipy.integrate as integrate
from scipy.linalg import matmul_toeplitz
from scipy.special import eval_legendre, jv, spherical_jn, sph_harm

from scipy.fft import fft as fft
# from fle_3d import FLEBasis3D
import matplotlib.pyplot as plt

from fast_volume_align_ot.fle_3d import FLEBasis3D


def test_nufft(nz,N):

    z = np.linspace(-1, 1, nz, endpoint=False)
    t = np.complex128(np.array([1,2]))
    
    s = np.zeros((N+1,),dtype=np.complex128)
    epsnuf = 1e-7
    nufft_type = 2
    plan = finufft.Plan(
        nufft_type,
        [nz],
        n_trans=1,#t.shape[1]*t.shape[2],#1,#
        eps=epsnuf,
        isign=-1,
        dtype=np.complex128,
    )

    Ns = (2*np.arange(0,N+1,dtype=np.float64)+1)/8
    plan.setpts(2*np.pi*Ns*2/nz)

    if nz%2 == 1:
        tmp = np.exp(2*np.pi*1j*z/(nz)).reshape(-1,)
        s = plan.execute(tmp*t)*(z[1]-z[0])
        s2 = np.zeros(s.shape,dtype=np.complex128)
        for i in range(len(s2)):
            s2[i] = np.sum(t*np.exp(-2*np.pi*1j*z*Ns[i])*(z[1]-z[0]))
        v = np.linalg.norm(s - s2)
        if v > 1e-2:
            print('--',v, np.linalg.norm(s2))
        
    else:
        s = plan.execute(t)*(z[1]-z[0])
        s2 = np.zeros(s.shape,dtype=np.complex128)
        for i in range(len(s2)):
            s2[i] = np.sum(t*np.exp(-2*np.pi*1j*z*Ns[i])*(z[1]-z[0]))
        v = np.linalg.norm(s - s2)
        if v > 1e-2:
            print('--',v, np.linalg.norm(s2))
    print(s,s2,s-s2)


def cramer(F,M,nz,N,max_l=20):
    fle = FLEBasis3D(M,M,1e-4,max_l=max_l,mode="complex",force_real="True")
    d = makedlk(fle)
    a = fle.evaluate_t(F)
    # a = 0*a
    # ind=29
    # a[ind] = 1
    epsnuf=1e-7
    t1 = time.time()
    z = np.linspace(-1, 1, nz, endpoint=False)
    #this equals np.arange(-(n-1)/2,(n-1)/2+1)*2/n - 1/n for n odd and
    #np.arange(-n//2,n//2)*2/n for n even
    a = a*d*fle.h
    t = fast_sum(np.real(a),z,fle) + 1j*fast_sum(np.imag(a),z,fle)#fast_sum(a,z,fle)#

    s = np.zeros((N+1,t.shape[1],t.shape[2]),dtype=np.complex128)
    #fft of t over the first axis, but we need to rescale the target points
    #set up NUFFT plan
    nufft_type = 2
    plan = finufft.Plan(
        nufft_type,
        [nz],
        n_trans=t.shape[1]*t.shape[2],#1,##1,#
        eps=epsnuf,
        isign=-1,
        dtype=np.complex128,
    )

    Ns = (2*np.arange(0,N+1,dtype=np.float64)+1)/8
    plan.setpts(2*np.pi*Ns*2/nz)


    # # plan.setpts(2*np.pi*2*Ns/nz*(nz-1)/(nz)) 
    # # #can also do better quadrature by doing 2 FFTs
    # # if nz%2 == 1:
    # #     b = b*np.exp(-1j*lmds/nz)
    s = plan.execute(t.reshape(t.shape[0],-1).T)*(z[1]-z[0])
    
    # # s = np.transpose(s.reshape(t.shape[1],t.shape[2],N+1),[2,0,1])

    s = np.reshape(s.T,(N+1,t.shape[1],t.shape[2]))

    # for l in range(s.shape[1]):
    #     for m in range(s.shape[2]):
    #         s[:,l,m] = plan.execute(t[:,l,m])*(z[1]-z[0])
    #         # # s2 = np.zeros(s[:,l,m].shape,dtype=np.complex128)
    #         # # for i in range(len(s2)):
    #         # #     s2[i] = np.sum(t[:,l,m]*np.exp(1j*z*Ns[i])*(z[1]-z[0]))
    #         # #     print('--',np.linalg.norm(s[:,l,m] - s2))
    #         # #     #this is not quite zero when doing
    #         # #     # a = 0*a
    #         # #     # a[1] = 1
    #         # #     #so something is off
    #         # if nz%2 == 1:
    #         #     tmp = np.exp(2*np.pi*1j*z/(nz)).reshape(-1,)
    #         #     s[:,l,m] = plan.execute(tmp*t[:,l,m])*(z[1]-z[0])
    #         #     s2 = np.zeros(s[:,l,m].shape,dtype=np.complex128)
    #         #     for i in range(len(s2)):
    #         #         s2[i] = np.sum(t[:,l,m]*np.exp(-2*np.pi*1j*z*Ns[i])*(z[1]-z[0]))
    #         #     v = np.linalg.norm(s[:,l,m] - s2)
    #         #     if v > 1e-2:
    #         #         print('--',l,m,v, np.linalg.norm(s2))
                
    #         # else:
    #         #     s[:,l,m] = plan.execute(t[:,l,m])*(z[1]-z[0])
    #         #     s2 = np.zeros(s[:,l,m].shape,dtype=np.complex128)
    #         #     for i in range(len(s2)):
    #         #         s2[i] = np.sum(t[:,l,m]*np.exp(-2*np.pi*1j*z*Ns[i])*(z[1]-z[0]))
    #         #     v = np.linalg.norm(s[:,l,m] - s2)
    #         #     if v > 1e-2:
    #         #         print('--',l,m,v, np.linalg.norm(s2))
    for n in range(s.shape[0]):
        s[n,:,:] =  s[n,:,:]*2/(2*(2*n+1)*np.pi)
        
    print(time.time()-t1)

    return s

def fast_sum(a,z,fle):
    #NOTE: assumes a is real. Run fast_sum(np.real(a)) + 1j*fast_sum(np.imag(a)) if a is complex
    nz = len(z)
    epsnuf=1e-7
    t = np.zeros((nz,fle.lmax+1,2*fle.lmax+1),dtype=np.complex128)
    #set up NUFFT plan
    nufft_type = 1
    plan = finufft.Plan(
        nufft_type,
        [nz],
        n_trans=1,
        eps=epsnuf,
        isign=1,
        dtype=np.complex128,
    )

 
    for l in range(fle.lmax+1):
        P = eval_legendre(l,z+1/(nz))

        # P = eval_legendre(l,z+1/(nz-1))
        # P2 = eval_legendre(l,z2)
        for m in range(-l,l+1):
            #run NUFFT step
            md=2*np.abs(m)-(m<0)
            inds = fle.idlm_list[l][md]
            #all the indices i which correspond to the current l and m
            lmds = fle.lmds[inds]
            
            plan.setpts(lmds*2/nz)
            b = a[inds]*np.exp(-1j*lmds+1j*lmds*1/(nz))/lmds

            #we want \sum_{j=1}^{r-1} exp( i*lmd*(z_j-z_r)) P(z_j)
            # so the above sets z_j = (z+1/(nz))_j.
            # For nz odd, this is np.arange(-n//2+1,n//2+1)*2/n 
            #is this what happens to the term exp( i*lmd*(z_j-z_r))?
            # we have (z_j-z_r) for j = 1, ... , r-1
            # defined on (-2, 0) for odd nz by
            # np.arange(-n//2+1,n//2+1)*2/n -1 - 1/n.
            #what does the +1j*lmds*1/(nz) do?
            #it shifts everything by 2/nz. That should be because of the flipping
            # and the setting of M[0,0] to 0, so P[0] does not matter
            # and even nz by
            if nz%2 == 0:
                b = b*np.exp(2*1j*lmds/nz)
            # these ones above work well, but I do not understand why
            
            f1=plan.execute(b)
            f = np.imag(f1[::-1])       #gives midpoint rule
            
            #Toeplitz step
            #not sure if the scipy implementation is the fastest...
            f[0] = 0


            I = 2*2*(1j)**(l)/4/(nz)*matmul_toeplitz((f,np.zeros(len(f))),P,workers=-1) 

            mind = 2*abs(m) - (m < 0)
            t[:,l,mind] = 1/6*(6*I)

    return t

### WORKS WELL
# def fast_sum(a,z,fle):
#     #NOTE: assumes a is real. Run step1(np.real(a)) + 1j*step1(np.imag(a)) if a is complex
#     #^actually. no longer sure about that. You call it with complex b and b2... so maybe just a typo?
#     nz = len(z)
#     z2 = np.concatenate([z, np.array([1])])
#     epsnuf=1e-7
#     t = np.zeros((nz,fle.lmax+1,2*fle.lmax+1),dtype=np.complex128)
#     #set up NUFFT plan
#     nufft_type = 1
#     plan = finufft.Plan(
#         nufft_type,
#         [nz],
#         n_trans=1,
#         eps=epsnuf,
#         isign=1,
#         dtype=np.complex128,
#     )
#     # plan2 = finufft.Plan(
#     #     nufft_type,
#     #     [nz+1],
#     #     n_trans=1,
#     #     eps=epsnuf,
#     #     isign=1,
#     #     dtype=np.complex128,
#     # )
 
#     for l in range(fle.lmax+1):
#         P = eval_legendre(l,z+1/(nz))

#         # P = eval_legendre(l,z+1/(nz-1))
#         # P2 = eval_legendre(l,z2)
#         for m in range(-l,l+1):
#             #run NUFFT step
#             md=2*np.abs(m)-(m<0)
#             inds = fle.idlm_list[l][md]
#             #all the indices i which correspond to the current l and m
#             lmds = fle.lmds[inds]
#             plan.setpts(lmds*2/nz)
#             # plan2.setpts(lmds*2/(nz+1))
#             b = a[inds]*np.exp(-1j*lmds+1j*lmds*1/(nz))/lmds

#             # b = a[inds]*np.exp(-1j*lmds+1j*lmds*1/(nz-1))/lmds
#             # b2 = a[inds]*np.exp(-1j*lmds)/lmds
#             f1=plan.execute(b)
#             # f2=plan2.execute(b2)

#             # f3 = np.imag(f2[1:][::-1])  #gives right Riemann sum
#             # f2 = np.imag(f2[:-1][::-1]) #gives left Riemann sum
#             f = np.imag(f1[::-1])       #gives midpoint rule
            
#             #Toeplitz step
#             #not sure if the scipy implementation is the fastest...
#             f[0] = 0
#             # f2[0] = 0
#             # f3[0] = 0

#             I = 2*2*(1j)**(l)/4/(nz)*matmul_toeplitz((f,np.zeros(len(f))),P,workers=-1) 
#             # I2 = 2*2*(1j)**(l)/4/(nz-1)*matmul_toeplitz((f2,np.zeros(len(f2))),P2[:-1]) 
#             # I3 = 2*2*(1j)**(l)/4/(nz-1)*matmul_toeplitz((f3,np.zeros(len(f3))),P2[1:]) 
            
#             mind = 2*abs(m) - (m < 0)
#             t[:,l,mind] = 1/6*(6*I)

#     return t


# def fast_sum(a,z,fle):
#     #NOTE: assumes a is real. Run step1(np.real(a)) + 1j*step1(np.imag(a)) if a is complex
#     #^actually. no longer sure about that. You call it with complex b and b2... so maybe just a typo?
#     nz = len(z)
#     z2 = np.concatenate([z, np.array([1])])
#     epsnuf=1e-7
#     t = np.zeros((nz,fle.lmax+1,2*fle.lmax+1),dtype=np.complex128)
#     #set up NUFFT plan
#     nufft_type = 1
#     plan = finufft.Plan(
#         nufft_type,
#         [nz],
#         n_trans=1,
#         eps=epsnuf,
#         isign=1,
#         dtype=np.complex128,
#     )
#     plan2 = finufft.Plan(
#         nufft_type,
#         [nz+1],
#         n_trans=1,
#         eps=epsnuf,
#         isign=1,
#         dtype=np.complex128,
#     )

        
#     for l in range(fle.lmax+1):
#         P = eval_legendre(l,z)
#         P2 = eval_legendre(l,z2)
#         for m in range(-l,l+1):
#             #run NUFFT step
#             md=2*np.abs(m)-(m<0)
#             inds = fle.idlm_list[l][md]
#             #all the indices i which correspond to the current l and m
#             lmds = fle.lmds[inds]
#             #for fixed l, we want to compute
#             # \sum_k 2*a_k/lambda_{lk} exp( 1j*lambda_{lk}*zj)
#             #for each j, where zj are quadrature points in -1, 1.
#             #We want to have all points except for left endpoint
#             #because we then flip them to get everything except
#             #right endpoint to have a left Riemann sum
#             # if nz%2 == 0:
#             #     plan.setpts(lmds*2/nz)
#             # else:
#             #     plan.setpts(lmds*2/(nz-1))
#             plan.setpts(lmds*2/nz)

#             b = np.complex128(a[inds]/(2*lmds))
#             if nz%2 == 1:
#                 b = b*np.exp(-1j*lmds/nz)
#             # b = b*np.exp(1j*lmds/nz)
#             f1=plan.execute(b)

#             f2 = np.zeros(f1.shape,dtype=np.complex128)
#             for i in range(len(f2)):
#                 f2[i] = np.sum(a[inds]/(2*lmds)*np.exp(1j*lmds*z[i]))
#             if np.linalg.norm(a[inds]) > 1e-3:
#                 print(f1[1:3])
#                 print(f2[1:3])
#                 print(np.linalg.norm(f1-f2))
#                 # plt.plot(np.abs(f1))
#                 # plt.plot(np.abs(f2))
#                 # plt.show()
#             # f = np.imag(f1[::-1])       #gives midpoint rule
#             f = np.imag(f1)       #gives midpoint rule

#             #Toeplitz step
#             #not sure if the scipy implementation is the fastest...
#             f[0] = 0

#             I = 2*(1j)**(l)/(nz)*matmul_toeplitz((f,np.zeros(len(f))),P,workers=-1) 

            
#             mind = 2*abs(m) - (m < 0)
#             t[:,l,mind] = I

#     return t

    
def cramer_brute_force(F,M,N,max_l=20):
    fle = FLEBasis3D(M,M,1e-4,max_l=max_l,mode="complex",force_real="True")
    a = fle.evaluate_t(F)
    d = makedlk(fle)
    # a = 0*a
    # ind=29
    # a[ind] = 1
    # print(fle.ks[ind], fle.ls[ind], fle.ms[ind])
    a = a*d*fle.h
    t1 = time.time()

    tensor_a = np.zeros((fle.kmax,max_l,2*max_l-1),dtype=np.complex128)
    lmds = np.zeros((fle.kmax,max_l),dtype=np.complex128)
    for i in range(len(a)):
        k = fle.ks[i]
        l = fle.ls[i]
        m = fle.ms[i]
        md = 2*np.abs(m) - (m < 0)
        tensor_a[k-1,l,md] = a[i]
        lmds[k-1,l] = fle.lmds[i]

    summed = np.zeros((N+1,tensor_a.shape[1],tensor_a.shape[2]),dtype=np.complex128)

    for n in range(summed.shape[0]):
        r = (2*n+1)/8
        for k in range(tensor_a.shape[0]):
            for l in range(tensor_a.shape[1]):
                for m in range(tensor_a.shape[2]):
                    lam = lmds[k,l]
                    summed[n,l,m] += tensor_a[k,l,m]*radial(r,l,lam)*2/(2*(2*n+1)*np.pi)
    print(time.time()-t1)
    return summed

def radial(r,l,lam):
    return spherical_jn(l, 2*np.pi*r) / ( (2*np.pi*r)**2 - lam**2)

    

def projs(F,M,nz,n_theta,n_phi,max_l=20):

    fle = FLEBasis3D(M,M,1e-6,max_l=max_l,mode="complex",force_real="True")
    d = makedlk(fle)
    
    # a = fle.expand(np.swapaxes(F,0,1))
    # a = fle.evaluate_t(np.swapaxes(F,0,1))
    a = fle.evaluate_t(F)

    # z = np.linspace(-1,1,nz)
    # R = M // 2
    # z = np.arange(-R, R + M % 2)/R
    z = np.linspace(-(1 - (1/nz)), 1 - (1/nz), nz, endpoint=True)
    a = a*d
    t = step1(np.real(a),z,fle) + 1j*step1(np.imag(a),z,fle)

    P = -step2(t,n_theta,n_phi,fle)*fle.h
    
    return np.transpose(P,(2,0,1))#make array have same convention as the radon transform


def makedlk(fle):
    d = -4*np.sqrt(2)*np.pi*(-1j)**fle.ls * \
    fle.lmds*(-1)**fle.ks
    return d


def step1(a,z,fle):
    #NOTE: assumes a is real. Run step1(np.real(a)) + 1j*step1(np.imag(a)) if a is complex
    #^actually. no longer sure about that. You call it with complex b and b2... so maybe just a typo?
    nz = len(z)
    z2 = np.concatenate([z, np.array([1])])
    epsnuf=1e-4
    t = np.zeros((nz,fle.lmax+1,2*fle.lmax+1),dtype=np.complex128)
    #set up NUFFT plan
    nufft_type = 1
    plan = finufft.Plan(
        nufft_type,
        [nz],
        n_trans=1,
        eps=epsnuf,
        isign=1,
        dtype=np.complex128,
    )
    plan2 = finufft.Plan(
        nufft_type,
        [nz+1],
        n_trans=1,
        eps=epsnuf,
        isign=1,
        dtype=np.complex128,
    )

        
    for l in range(fle.lmax+1):
        P = eval_legendre(l,z+1/(nz-1))
        P2 = eval_legendre(l,z2)
        for m in range(-l,l+1):
            #run NUFFT step
            md=2*np.abs(m)-(m<0)
            inds = fle.idlm_list[l][md]
            #all the indices i which correspond to the current l and m
            lmds = fle.lmds[inds]
            plan.setpts(lmds*2/nz)
            plan2.setpts(lmds*2/(nz+1))
            b = a[inds]*np.exp(-1j*lmds+1j*lmds*1/(nz-1))/lmds
            b2 = a[inds]*np.exp(-1j*lmds)/lmds
            f1=plan.execute(b)
            f2=plan2.execute(b2)

            f3 = np.imag(f2[1:][::-1])  #gives right Riemann sum
            f2 = np.imag(f2[:-1][::-1]) #gives left Riemann sum
            f = np.imag(f1[::-1])       #gives midpoint rule
            #are you zero-stacking in the right way?
            #The superdiagonals should be zero...
            
            #Toeplitz step
            #not sure if the scipy implementation is the fastest...
            f[0] = 0
            f2[0] = 0
            f3[0] = 0

            I = 2*2*(1j)**(l)/4/(nz-1)*matmul_toeplitz((f,np.zeros(len(f))),P,workers=-1) 
            I2 = 2*2*(1j)**(l)/4/(nz-1)*matmul_toeplitz((f2,np.zeros(len(f2))),P2[:-1]) 
            I3 = 2*2*(1j)**(l)/4/(nz-1)*matmul_toeplitz((f3,np.zeros(len(f3))),P2[1:]) 
            
            mind = 2*abs(m) - (m < 0)
            t[:,l,mind] = 1/6*(4*I+I2+I3)

    return t


#before march 20, 2026
# def step1(a,z,fle):
#     #NOTE: assumes a is real. Run step1(np.real(a)) + 1j*step1(np.imag(a)) if a is complex
#     #^actually. no longer sure about that
#     nz = len(z)
#     z2 = np.concatenate([z, np.array([1])])
#     epsnuf=1e-4
#     t = np.zeros((nz,fle.lmax+1,2*fle.lmax+1),dtype=np.complex128)
#     #set up NUFFT plan
#     nufft_type = 1
#     plan = finufft.Plan(
#         nufft_type,
#         [nz],
#         n_trans=1,
#         eps=epsnuf,
#         isign=1,
#         dtype=np.complex128,
#     )
#     plan2 = finufft.Plan(
#         nufft_type,
#         [nz+1],
#         n_trans=1,
#         eps=epsnuf,
#         isign=1,
#         dtype=np.complex128,
#     )

        
#     for l in range(fle.lmax+1):
#         P = eval_legendre(l,z+1/(nz-1))
#         P2 = eval_legendre(l,z2)
#         for m in range(-l,l+1):
#             #run NUFFT step
#             md=2*np.abs(m)-(m<0)
#             inds = fle.idlm_list[l][md]
#             #all the indices i, which correspond to the current l and m
#             lmds = fle.lmds[inds]
#             plan.setpts(lmds*2/nz)
#             plan2.setpts(lmds*2/(nz+1))
#             b = a[inds]*np.exp(-1j*lmds+1j*lmds*1/(nz-1))/lmds
#             b2 = a[inds]*np.exp(-1j*lmds)/lmds
#             f1=plan.execute(b)
#             f2=plan2.execute(b2)

#             f3 = np.imag(f2[1:][::-1])  #gives right Riemann sum
#             f2 = np.imag(f2[:-1][::-1]) #gives left Riemann sum
#             f = np.imag(f1[::-1])       #gives midpoint rule

#             #Toeplitz step
#             #not sure if the scipy implementation is the fastest...
#             f[0] = 0
#             f2[0] = 0
#             f3[0] = 0

#             I = 2*2*(1j)**(l)/4/(nz-1)*matmul_toeplitz((f,np.zeros(len(f))),P,workers=-1) 
#             I2 = 2*2*(1j)**(l)/4/(nz-1)*matmul_toeplitz((f2,np.zeros(len(f2))),P2[:-1]) 
#             I3 = 2*2*(1j)**(l)/4/(nz-1)*matmul_toeplitz((f3,np.zeros(len(f3))),P2[1:]) 
            
#             mind = 2*abs(m) - (m < 0)
#             t[:,l,mind] = 1/6*(4*I+I2+I3)

#     return t


def step2(t,n_theta,n_phi,fle):
    lmax = fle.lmax
    lmax = t.shape[1]-1
    grid = "equiangular"
    
    isht = th.InverseRealSHT(
        n_theta,
        n_phi,
        lmax=lmax + 1,
        mmax=lmax + 1,
        grid=grid,
        csphase=True,
    )
    t1, t2 = fle.torch_reshape_order(t)
    t1 = torch.tensor(t1, dtype=torch.complex128)
    t2 = torch.tensor(t2, dtype=torch.complex128)

    P1 = np.complex128(isht(t1.to("cpu")).cpu().numpy())
    P2 = np.complex128(isht(t2.to("cpu")).cpu().numpy())
    return (P1 + 1j * P2)