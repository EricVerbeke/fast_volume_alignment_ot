import numpy as np
import matplotlib.pyplot as plt


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


def orthogonal_2D_projection_plot(volume, cmap='viridis', size=(7,7)):
    nz, ny, nx = volume.shape
    
    fig, axs = plt.subplots(1, 3, figsize=size)
    
    p_z = axs[0].imshow(np.sum(volume, axis=0), cmap)
    axs[0].axis('off')
    p_y = axs[1].imshow(np.sum(volume, axis=1), cmap)
    axs[1].axis('off')
    p_x = axs[2].imshow(np.sum(volume, axis=2), cmap)
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()