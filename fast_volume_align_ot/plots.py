import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def orthogonal_2D_slice_plot(volume, cmap='viridis', size=(7,7)):
    
    nz, ny, nx = volume.shape
    
    fig, axs = plt.subplots(1, 3, figsize=size)
    
    s_z = axs[0].imshow(volume[nz//2, :, :], cmap)
    axs[0].axis('off')
    s_y = axs[1].imshow(volume[:, ny//2, :], cmap)
    axs[1].axis('off')
    s_x = axs[2].imshow(volume[:, :, nx//2], cmap)
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def orthogonal_2D_projection_plot(volume, cmap='viridis', size=(7,7), title=True):
    
    nz, ny, nx = volume.shape
    
    fig, axs = plt.subplots(1, 3, figsize=size)
    
    p_z = axs[0].imshow(np.sum(volume, axis=0), cmap)
    axs[0].axis('off')
    p_y = axs[1].imshow(np.sum(volume, axis=1), cmap)
    axs[1].axis('off')
    p_x = axs[2].imshow(np.sum(volume, axis=2), cmap)
    axs[2].axis('off')

    if title:
        axs[0].set_title('axis=0, xy-plane', size=10)
        axs[1].set_title('axis=1, xz-plane', size=10)
        axs[2].set_title('axis=2, yz-plane', size=10)
    
    plt.tight_layout()
    plt.show()


def rotational_distance_heatmap(dists, ticks='degrees', plot_min=True):

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
        ax.set_yticks(np.linspace(0, n_theta-1, 9)[:-1])
        ax.set_yticklabels(np.arange(0, 360, 45), rotation=45, size=8)
        ax.tick_params(axis="both", which="major", pad=0)
    
    if ticks == 'radians':
        ax.set_xticks(np.linspace(0, n_phi-1, 9)[:-1])
        ax.set_xticklabels([0, r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'],rotation=0, size=8)
        ax.set_yticks(np.linspace(0, n_theta-1, 9)[:-1])
        ax.set_yticklabels([0, r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'],rotation=0, size=8)
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