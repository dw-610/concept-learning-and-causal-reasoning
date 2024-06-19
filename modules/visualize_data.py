"""
This module contains functions for visualizing the data.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

def show_image(image: np.ndarray):
    plt.imshow(image)
    plt.show()

# ------------------------------------------------------------------------------

def show_processed_image(image: np.ndarray):
    plt.imshow(image/2.0 + 0.5)
    plt.show()

# ------------------------------------------------------------------------------
    
def show_processed_images(images: np.ndarray, labels: np.ndarray, n: int):
    num = n**2
    _, axes = plt.subplots(n, n)
    for i in range(num):
        row = i//n
        col = i%n
        ax = axes[row, col]
        ax.imshow(images[i]/2.0 + 0.5)
        ax.set_title(labels[i])
        ax.axis('off')
    plt.show()

# ------------------------------------------------------------------------------