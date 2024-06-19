"""
This module contains functions to take the raw data to the data ready for
domain learning.
"""

# ------------------------------------------------------------------------------
# imports

import os
import numpy as np
import pandas as pd
from PIL import Image

import modules.visualize_data as vis

# ------------------------------------------------------------------------------

def load_image(path: str):
    """
    Returns a PIL image object.
    """
    return Image.open(path)

# ------------------------------------------------------------------------------

def resize_image(image: np.ndarray, size: int):
    """
    Resize the image to the given size. Resizes the largest dimension to the
    given size to preserve the aspect ratio.

    Returns a uint8 numpy array.
    """
    width = image.size[0]
    height = image.size[1]
    if width > height:
        new_width = size
        new_height = int(height/width*size)
    else:
        new_height = size
        new_width = int(width/height*size)
    image = image.resize((new_width, new_height))
    return np.array(image)

# ------------------------------------------------------------------------------

def process_image(image: np.ndarray):
    """
    Process the image.

    Returns a float32 numpy array where the values are between -1 and 1.
    """
    image = np.array(image).astype(np.float32)
    processed = image/255.0*2.0 - 1.0
    return processed

# ------------------------------------------------------------------------------

def pad_image(image: np.ndarray):
    """
    Pads the image to make it square.
    """
    width = image.shape[0]
    height = image.shape[1]

    if width == height:
        return image

    if width > height:
        if (width - height) % 2 == 0:
            pad = (width - height) // 2
            padded = np.pad(image, ((0,0), (pad, pad), (0,0)), mode='constant')
        else:
            pad = (width - height) // 2
            padded = np.pad(image, ((0,0), (pad, pad+1), (0,0)), mode='constant')
    else:
        if (height - width) % 2 == 0:
            pad = (height - width) // 2
            padded = np.pad(image, ((pad, pad), (0,0), (0,0)), mode='constant')
        else:
            pad = (height - width) // 2
            padded = np.pad(image, ((pad, pad+1), (0,0), (0,0)), mode='constant')
    return padded

# ------------------------------------------------------------------------------

def load_images(dir: str, im_size: int, verbose: bool = True):
    """
    Load all the images in the directory, returns array that is type uint8.
    """
    files = os.listdir(dir)
    files = list(np.sort(files))
    images = np.empty((len(files), im_size, im_size, 3), dtype=np.float32)
    if verbose:
        print(f'\nLoading images from {dir}')
    for i, file in enumerate(files):
        image = load_image(dir + file)
        image = resize_image(image, im_size)
        image = process_image(image)
        image = pad_image(image)
        images[i] = image
        if verbose:
            print(f'\r{i+1}/{len(files)}', end='')
    if verbose:
        print(' -> Done!')
    return images

# ------------------------------------------------------------------------------

def load_labels(file: str):
    """
    Load the labels from the given file.
    """
    labels = pd.read_csv(file, delimiter=';')
    return labels['ClassId'].values

# ------------------------------------------------------------------------------

def labels_to_one_hot(labels: np.ndarray):
    """
    Convert the labels to one-hot encoding.
    """
    n_classes = np.max(labels) + 1
    one_hot = np.zeros((len(labels), n_classes)).astype(np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot

# ------------------------------------------------------------------------------

def shuffle_data(images: np.ndarray, labels: np.ndarray):
    """
    Shuffle the data.
    """
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    return shuffled_images, shuffled_labels

# ------------------------------------------------------------------------------

def load_memmap_labels(file: str, shape: tuple):
    """
    Load the labels from the given file.
    """
    labels = np.memmap(file, dtype='float32', mode='r', shape=shape)
    return labels

# ------------------------------------------------------------------------------

def map_labels(labels: np.ndarray, label_map: dict):
    """
    Map the labels to the new labels.

    Labels should be a 1D numpy array of integers.
    """
    new_labels = np.empty(labels.shape, dtype=np.int64)
    for i, label in enumerate(labels):
        new_labels[i] = label_map[label]
    return new_labels

# ------------------------------------------------------------------------------

def save_data(
        directory: str,
        training_images: np.ndarray,
        training_labels: np.ndarray,
        testing_images: np.ndarray,
        testing_labels: np.ndarray,
        test_number: int,
        identifier: str = ''
    ):
    """
    Saves the dataset as memmap arrays. All data should be of type float32.

    Splits the original test set into an actual test set (test_number samples)
    and an additional training set (the rest of the samples).

    Can add an identifier to the file name.
    """
    csl_trn_size = (testing_images.shape[0] - test_number, 
                    *testing_images.shape[1:])
    tst_size = (test_number, *testing_images.shape[1:])

    csl_trn_labels_size = (testing_labels.shape[0] - test_number, 
                           *testing_labels.shape[1:])
    tst_labels_size = (test_number, *csl_trn_labels_size[1:])

    test_idx = np.random.choice(
        np.arange(testing_images.shape[0]),
        test_number,
        replace=False
    )
    csl_trn_idx = np.setdiff1d(np.arange(testing_images.shape[0]), test_idx)

    if directory[-1] != '/':
        directory += '/'

    if identifier != '':
        identifier = '_' + identifier

    dom_trn_images = np.memmap(
        f'{directory}signs_dom_trn_data{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=training_images.shape
    )

    dom_trn_labels = np.memmap(
        f'{directory}signs_dom_trn_labels{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=training_labels.shape
    )

    csl_trn_images = np.memmap(
        f'{directory}signs_csl_trn_data{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=csl_trn_size
    )

    csl_trn_labels = np.memmap(
        f'{directory}signs_csl_trn_labels{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=csl_trn_labels_size
    )

    tst_images = np.memmap(
        f'{directory}signs_tst_data{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=tst_size
    )

    tst_labels = np.memmap(
        f'{directory}signs_tst_labels{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=tst_labels_size
    )

    dom_trn_images[:] = training_images
    dom_trn_labels[:] = training_labels
    csl_trn_images[:] = testing_images[csl_trn_idx]
    csl_trn_labels[:] = testing_labels[csl_trn_idx]
    tst_images[:] = testing_images[test_idx]
    tst_labels[:] = testing_labels[test_idx]

    dom_trn_images.flush()
    dom_trn_labels.flush()
    csl_trn_images.flush()
    csl_trn_labels.flush()
    tst_images.flush()
    tst_labels.flush()

    del dom_trn_images
    del dom_trn_labels
    del csl_trn_images
    del csl_trn_labels
    del tst_images
    del tst_labels

    print('Data saved.')

# ------------------------------------------------------------------------------
    
def save_new_labels(
        directory: str,
        domain_training_labels: np.ndarray,
        causal_training_labels: np.ndarray,
        testing_labels: np.ndarray,
        identifier: str = ''
    ):
    """
    Saves label arrays as memmap arrays. All data should be of type float32.

    Can add an identifier to the file name.
    """

    if directory[-1] != '/':
        directory += '/'

    if identifier != '':
        identifier = '_' + identifier

    dom_trn_size = domain_training_labels.shape
    csl_trn_size = causal_training_labels.shape
    tst_size = testing_labels.shape

    dom_trn_labels = np.memmap(
        f'{directory}signs_dom_trn_labels{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=dom_trn_size
    )

    csl_trn_labels = np.memmap(
        f'{directory}signs_csl_trn_labels{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=csl_trn_size
    )

    tst_labels = np.memmap(
        f'{directory}signs_tst_labels{identifier}.npy',
        dtype='float32',
        mode='w+',
        shape=tst_size
    )

    dom_trn_labels[:] = domain_training_labels
    csl_trn_labels[:] = causal_training_labels
    tst_labels[:] = testing_labels

    dom_trn_labels.flush()
    csl_trn_labels.flush()
    tst_labels.flush()

    del dom_trn_labels
    del csl_trn_labels
    del tst_labels

    print('Labels saved.')

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    im_size = 64
    images = load_images('raw_data/train/images/00003/', im_size)
    print(images.shape)
    print(images.dtype)
    # vis.show_processed_image(images[0])

    labels = load_labels('raw_data/train/anno/GT-00003.csv')
    one_hot = labels_to_one_hot(labels)
    print(labels.shape)
    print(one_hot.shape)

    breakpoint()

    vis.show_processed_images(images, labels, 10)