"""
This file contains miscellanious utility functions that are useful throughout
the learning process.
"""

# ------------------------------------------------------------------------------

import os
import sys
import numpy as np
import cv2

import tensorflow as tf

K = tf.keras.backend

# ------------------------------------------------------------------------------


def get_unused_name(
        file_path: str
    ):
    """
    Function for getting an unused file name at a specified location.

    Parameters
    ----------
    file_path : str
        Path (including file name) to where the file should be saved.

    Returns
    -------
    new_fname : str
        If the filename doesn't already exist, this is just file_path.
        If the filename does exist, returns file_path with a modified filename.
    """

    # if the name doesn't exist, return the given path
    if not os.path.exists(file_path):
        return file_path
    
    # if it does exist, modify the file name
    filename, file_extension = os.path.splitext(file_path)
    counter = 1
    new_fname = f"{filename}_{counter}{file_extension}"
    while os.path.exists(new_fname):
        counter += 1
        new_fname = f"{filename}_{counter}{file_extension}"
    return new_fname


# ------------------------------------------------------------------------------


def one_hot_to_ints(
        one_hot: np.ndarray
    ) -> np.ndarray:
    """
    Function for converting a one-hot encoded array to an array of integers.

    Parameters
    ----------
    one_hot : np.ndarray
        The one-hot encoded array.

    Returns
    -------
    ints : np.ndarray
        The array of integers.
    """

    ints = np.zeros(one_hot.shape[0])
    for i in range(one_hot.shape[0]):
        ints[i] = np.argmax(one_hot[i])
    return ints


# ------------------------------------------------------------------------------


def print_model_summary(model, input_shape):
    """
    Function to print a summary of a model, since model.summary() seems to 
    sometimes not work with custom layers.
    The compute_output_shape() method of each layer MUST be defined for each 
    layer of the model to use this function.

    Parameters
    ----------
    model : tf.keras.models.Model (or subclass of)
        The model to print a summary of.
    """

    print("_________________________________________________________________")
    print(f"                           {model.name}                          ")
    print("_________________________________________________________________")
    print(" Layer                       Output Shape              Param #   ")
    print("=================================================================")

    total_params = 0
    shape = input_shape

    for layer in model.layers:

        assert hasattr(layer, 'compute_output_shape'), \
            "compute_output_shape method is undefined forLayer {layer.name}"

        shape = layer.compute_output_shape(shape)

        layer_str = f" {layer.name}" + \
            f"{' ' * (28 - len(layer.name))}" + \
            f"{shape}{' ' * (26 - len(str(shape)))}" + \
            f"{layer.count_params()}"
        
        print(layer_str)

        total_params += layer.count_params()

    trainable_params = int(sum([K.count_params(w) for w \
                            in model.trainable_weights]))
    non_trainable_params = int(sum([K.count_params(w) for w \
                                in model.non_trainable_weights]))

    print("=================================================================")
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    print(f"Non-trainable params: {non_trainable_params}")
    print("_________________________________________________________________\n")


# ------------------------------------------------------------------------------