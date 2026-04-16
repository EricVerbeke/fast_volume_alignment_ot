import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def image_plot(f, c='viridis', cbar=True):
    
    plt.imshow(f, cmap=c)
    if cbar:
        plt.colorbar()
    plt.show()


def two_imshow(image1, image2, cmap='viridis', size=(7,7), cbar=False):
    
    fig, axs = plt.subplots(1, 2, figsize=size)

    img1 = axs[0].imshow(image1, cmap)
    axs[0].axis('off')

    img2 = axs[1].imshow(image2, cmap)
    axs[1].axis('off')
    
    if cbar:
        cbar1 = fig.colorbar(img1, ax=axs[0], aspect=size[0])
        cbar2 = fig.colorbar(img2, ax=axs[1], aspect=size[0])

    plt.tight_layout()
    plt.show()


def orthogonal_2D_slice_plot(volume, cmap='viridis', size=(7,7), title=True):
    
    nz, ny, nx = volume.shape
    
    fig, axs = plt.subplots(1, 3, figsize=size)
    
    s_z = axs[0].imshow(volume[nz//2, :, :], cmap)
    plt.colorbar(s_z,ax=axs[0],fraction=0.046, pad=0.04)
    axs[0].axis('off')
    s_y = axs[1].imshow(volume[:, ny//2, :], cmap)
    plt.colorbar(s_y,ax=axs[1],fraction=0.046, pad=0.04)
    axs[1].axis('off')
    s_x = axs[2].imshow(volume[:, :, nx//2], cmap)
    plt.colorbar(s_x,ax=axs[2],fraction=0.046, pad=0.04)
    axs[2].axis('off')

    if title:
        axs[0].set_title('axis=0 slice', size=10)
        axs[1].set_title('axis=1 slice', size=10)
        axs[2].set_title('axis=2 slice', size=10)
    
    plt.tight_layout()
    plt.show()


def orthogonal_2D_projection_plot(volume, cmap='viridis', size=(7,7), title=True):
    
    fig, axs = plt.subplots(1, 3, figsize=size)
    
    p_z = axs[0].imshow(np.sum(volume, axis=0), cmap)
    plt.colorbar(p_z,ax=axs[0],fraction=0.046, pad=0.04)
    axs[0].axis('off')
    p_y = axs[1].imshow(np.sum(volume, axis=1), cmap)
    plt.colorbar(p_z,ax=axs[1],fraction=0.046, pad=0.04)
    axs[1].axis('off')
    p_x = axs[2].imshow(np.sum(volume, axis=2), cmap)
    plt.colorbar(p_z,ax=axs[2],fraction=0.046, pad=0.04)
    axs[2].axis('off')

    if title:
        axs[0].set_title('projection axis=0', size=10)
        axs[1].set_title('projection axis=1', size=10)
        axs[2].set_title('projection axis=2', size=10)
    
    plt.tight_layout()
    plt.show()


def orthogonal_1D_projection_comparison_plot(f, Pf, size=(12,4)):

    n_phi, n_p, n_theta = Pf.shape

    ### these are how I think axis projections would be defined, but order doesn't match
    axis_0_proj = np.sum(f, axis=(1,2))  # x-axis
    axis_1_proj = np.sum(f, axis=(0,2))  # z-axis
    axis_2_proj = np.sum(f, axis=(0,1))  # y-axis
    
    x_axis_slice = Pf[0, :, n_theta//2]
    y_axis_slice = Pf[n_phi//4, :, n_theta//2]
    z_axis_slice = Pf[0, :, 0]
        
    ### EV: NOTE the order of these plots might need to change
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    axs[0].plot(axis_0_proj, color='blue', linewidth=2)
    axs[0].plot(x_axis_slice, linestyle='--', color='darkorange', linewidth=2)
    axs[0].set_title('x-axis', size=10)
    
    axs[1].plot(axis_1_proj, color='blue', linewidth=2)
    axs[1].plot(y_axis_slice, linestyle='--', color='darkorange', linewidth=2)
    axs[1].set_title('y-axis', size=10)
    
    axs[2].plot(axis_2_proj, color='blue', linewidth=2)
    axs[2].plot(z_axis_slice, linestyle='--', color='darkorange', linewidth=2)
    axs[2].set_title('z-axis', size=10)
    
    for ax in axs:
        ax.grid(which='major', linestyle='--')
        
    plt.tight_layout()
    plt.show()   


def rotational_distance_heatmap(dists, ticks='degrees', plot_min=True):
    """
    plot rotational distances in theta and phi
    update to take theta, phi as inputs? or just only label plot at specific angles
    """

    n_theta, n_phi = dists.shape
    
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(dists, cmap="magma")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=8, rotation=0)
    
    if ticks == 'degrees':
        ax.set_xticks(np.linspace(0, n_phi-1, 9)[:-1])
        ax.set_xticklabels(np.arange(0, 360, 45), rotation=45, size=8)
        ax.set_yticks(np.linspace(0, n_theta-1, 5))
        ax.set_yticklabels(np.arange(0, 180 + 1, 45), rotation=45, size=8)
        ax.tick_params(axis="both", which="major", pad=0)
    
    if ticks == 'radians':
        ax.set_xticks(np.linspace(0, n_phi-1, 9)[:-1])
        ax.set_xticklabels([0,r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$',r'$5\pi/4$',r'$3\pi/2$',r'$7\pi/4$'],rotation=0,size=8)
        ax.set_yticks(np.linspace(0, n_theta-1, 5))
        ax.set_yticklabels([0, r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'],rotation=0, size=8)
        ax.tick_params(axis="both", which="major", pad=0)  
    
    if plot_min:
        min_theta_idx, min_phi_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
        ax.axhline(y=min_theta_idx, color='cyan', linestyle='--', linewidth=2)
        ax.axvline(x=min_phi_idx, color='cyan', linestyle='--', linewidth=2)
        ax.plot(min_phi_idx, min_theta_idx, 'X', color='cyan', markersize=8, markeredgecolor='black')
    
    ax.set_ylabel(r'$\theta$', rotation=0, labelpad=8, fontsize=10)
    ax.set_xlabel(r'$\varphi$', rotation=0, labelpad=0, fontsize=10)
    ax.set_title('rotational distance heatmap', size=10, pad=0)
    plt.show() 