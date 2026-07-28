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


def orthogonal_2D_slice_plot(volume, cmap='viridis', size=(7,7), title=True, cbar=True):
    
    nz, ny, nx = volume.shape
    
    fig, axs = plt.subplots(1, 3, figsize=size)
    
    s_z = axs[0].imshow(volume[nz//2, :, :], cmap)
    axs[0].axis('off')
    s_y = axs[1].imshow(volume[:, ny//2, :], cmap)
    axs[1].axis('off')
    s_x = axs[2].imshow(volume[:, :, nx//2], cmap)
    axs[2].axis('off')

    if title:
        axs[0].set_title('slice axis=0', size=10)
        axs[1].set_title('slice axis=1', size=10)
        axs[2].set_title('slice axis=2', size=10)

    if cbar:
        plt.colorbar(s_z, ax=axs[0], fraction=0.046, pad=0.04)
        plt.colorbar(s_y, ax=axs[1], fraction=0.046, pad=0.04)
        plt.colorbar(s_x, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()


def orthogonal_2D_projection_plot(volume, cmap='viridis', size=(7,7), title=True, cbar=True):
    
    fig, axs = plt.subplots(1, 3, figsize=size)
    
    p_z = axs[0].imshow(np.sum(volume, axis=0), cmap)
    axs[0].axis('off')
    p_y = axs[1].imshow(np.sum(volume, axis=1), cmap)
    axs[1].axis('off')
    p_x = axs[2].imshow(np.sum(volume, axis=2), cmap)
    axs[2].axis('off')

    if title:
        axs[0].set_title('projection axis=0', size=10)
        axs[1].set_title('projection axis=1', size=10)
        axs[2].set_title('projection axis=2', size=10)

    if cbar:
        plt.colorbar(p_z,ax=axs[0], fraction=0.046, pad=0.04)
        plt.colorbar(p_z,ax=axs[1], fraction=0.046, pad=0.04)
        plt.colorbar(p_z,ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()