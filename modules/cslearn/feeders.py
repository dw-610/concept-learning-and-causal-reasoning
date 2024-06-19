"""
This file contains generator classes that can be passed to the model.fit() 
method in keras.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf
import numpy as np
from typing import Optional

# ------------------------------------------------------------------------------


class ClassifierFeederFromArray(tf.data.Dataset):
    """
    This class creates an object for feeding data to a model for training. It
    can be used with the model.fit() method in keras. This class takes in a
    numpy array containing the data, and a numpy array containing the labels.
    """
    def __new__(
            cls,
            is_training: bool,
            data_array: np.ndarray,
            label_array: np.ndarray,
            batch_size: int = 32
        ):
        """
        Constructor method for the ClassifierFeederFromArray class.


        Parameters
        ----------
        is_training : bool
            A boolean indicating whether the data is for training or validation.
        data_array : numpy.array, optional
            Numpy array containing the data.
            Shape is (n_samples, height, width, channels).
            If the channels dimension is not present, a value of 1 is assumed.
            Values should be in the range 0-255.
        label_array : numpy.array, optional
            Numpy array containing the labels.
            Shape is (n_samples,), with integer-valued labels indicating the
            class of the sample in the range 0 to n_classes-1.
        batch_size : int, optional
            The size of the batch loaded from the generator.
            Default value is 32.

        Returns
        -------
        tf.data.Dataset
            A tf.data.Dataset object containing the data and labels.
        """

        assert len(data_array.shape) == 3 or len(data_array.shape) == 4, \
            "Data array must have 3 or 4 dimensions."
        assert isinstance(data_array, np.ndarray), \
            "Data array must be a numpy array."
        assert isinstance(label_array, np.ndarray), \
            "Label array must be a numpy array."
        assert data_array.shape[0] == label_array.shape[0], \
            "Data and label arrays must have the same number of samples."

        # set constants
        n_samples = data_array.shape[0]
        n_classes = len(np.unique(label_array))

        # convert label array to 1D if necessary
        if len(label_array.shape) > 1:
            label_array = np.reshape(label_array, newshape=(n_samples,))

        # preprocessing
        data_array = data_array.astype(np.float32)/255.0 # normalize to 0-1
        data_array = (data_array - 0.5)/0.5 # normalize to -1 to 1
        if len(data_array.shape) == 3:
            data_array = np.expand_dims(data_array,axis=-1)
        one_hot_array = np.zeros(shape=(n_samples,n_classes)) # one-hot encode labels
        one_hot_array[np.arange(n_samples),label_array] = 1.0
        one_hot_array = one_hot_array.astype(np.float32)

        # convert to tf.data.Dataset object
        ds = tf.data.Dataset.from_tensor_slices((data_array,one_hot_array))
        if is_training:
            ds = ds.repeat()
            ds = ds.shuffle(buffer_size=n_samples)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # set sharding to DATA
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)

        return ds
    

# ------------------------------------------------------------------------------
    

class ClassifierFeederFromMemmap(tf.data.Dataset):
    def __new__(
            cls,
            is_training: bool,
            data_file: str,
            label_file: str,
            data_array_shape: tuple,
            label_array_shape: tuple,
            batch_size: int = 32,
            buffer_size: Optional[int] = 10000
        ):
        """
        Constructor method for the ClassifierFeederFromMemmap class.

        Parameters
        ----------
        is_training : bool
            A boolean indicating whether the data is for training or validation.
        data_file : str
            Path to the .npy file containing the data.
            Should be saved as type float32.
        label_file : str
            Path to the .npy file containing the labels.
            Should be saved as type float32.
        data_array_shape : tuple
            Shape of the data array.
            Should have 4 dimensions: (n_samples, height, width, channels).
        label_array_shape : tuple
            Shape of the label array.
            Should have 2 dimensions: (n_samples, n_classes).
        batch_size : int, optional
            The size of the batch loaded from the generator.
            Default value is 32.
        buffer_size : int, optional
            The size of the buffer used for shuffling.
            Default value is 10000.

        Returns
        -------
        tf.data.Dataset
            A tf.data.Dataset object containing the data and labels.
        """

        assert data_array_shape[0] == label_array_shape[0], \
            "Data and label arrays must have the same number of samples."
        
        n_classes = label_array_shape[1]

        height = data_array_shape[1]
        width = data_array_shape[2]
        channels = data_array_shape[3]

        # define generator function
        def data_gen(data_file, label_file, data_shape, label_shape):
            data = np.memmap(
                data_file,
                dtype='float32',
                mode='r',
                shape=tuple(data_shape)
            )
            labels = np.memmap(
                label_file,
                dtype='float32',
                mode='r',
                shape=tuple(label_shape)
            )
            for i in range(data_shape[0]):
                yield data[i], labels[i]

        # convert to tf.data.Dataset object
        ds = tf.data.Dataset.from_generator(
            generator=data_gen,
            args=[data_file, label_file, data_array_shape, label_array_shape],
            output_signature=(
                tf.TensorSpec(shape=(height,width,channels), dtype=tf.float32),
                tf.TensorSpec(shape=(n_classes,), dtype=tf.float32)
            )
        )
        if is_training:
            ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.repeat()
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # set sharding to DATA
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)

        return ds
        
        
# ------------------------------------------------------------------------------
    

class AutoencoderFeederFromArray(tf.data.Dataset):
    """
    This class creates an object for feeding data to a model for training. It
    can be used with the model.fit() method in keras. This class takes in a
    numpy array containing the data, and a numpy array containing the labels.
    """
    def __new__(
            cls,
            is_training: bool,
            data_array: np.ndarray,
            batch_size: int = 32
        ):
        """
        Constructor method for the AutoencoderFeederFromArray class.

        Parameters
        ----------
        is_training : bool
            A boolean indicating whether the data is for training or validation.
        data_array : numpy.array, optional
            Numpy array containing the data.
            Shape is (n_samples, height, width, channels).
            If the channels dimension is not present, a value of 1 is assumed.
            Values should be in the range 0-255.
        batch_size : int, optional
            The size of the batch loaded from the generator.
            Default value is 32.

        Returns
        -------
        tf.data.Dataset
            A tf.data.Dataset object containing the data and labels.
        """

        assert len(data_array.shape) == 3 or len(data_array.shape) == 4, \
            "Data array must have 3 or 4 dimensions."
        assert isinstance(data_array, np.ndarray), \
            "Data array must be a numpy array."

        # set constants
        n_samples = data_array.shape[0]

        # preprocessing
        data_array = data_array.astype(np.float32)/255.0 # normalize to 0-1
        data_array = (data_array - 0.5)/0.5 # normalize to -1 to 1
        if len(data_array.shape) == 3:
            data_array = np.expand_dims(data_array,axis=-1)

        # convert to tf.data.Dataset object
        ds = tf.data.Dataset.from_tensor_slices((data_array,data_array))
        if is_training:
            ds = ds.repeat()
            ds = ds.shuffle(buffer_size=n_samples)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # set sharding to DATA
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)

        return ds
    

# ------------------------------------------------------------------------------
    

class AutoencoderFeederFromMemmap(tf.data.Dataset):
    def __new__(
            cls,
            is_training: bool,
            data_file: str,
            data_array_shape: tuple,
            batch_size: int = 32,
            buffer_size: Optional[int] = 10000
        ):
        """
        Constructor method for the AutoencoderFeederFromMemmap class.

        Parameters
        ----------
        is_training : bool
            A boolean indicating whether the data is for training or validation.
        data_file : str
            Path to the .npy file containing the data.
            Should be saved as type float32.
        data_array_shape : tuple
            Shape of the data array.
            Should have 4 dimensions: (n_samples, height, width, channels).
        batch_size : int, optional
            The size of the batch loaded from the generator.
            Default value is 32.
        buffer_size : int, optional
            The size of the buffer used for shuffling.
            Default value is 10000.

        Returns
        -------
        tf.data.Dataset
            A tf.data.Dataset object containing the data and labels.
        """

        height = data_array_shape[1]
        width = data_array_shape[2]
        channels = data_array_shape[3]

        # define generator function
        def data_gen(data_file, data_shape):
            data_array = np.memmap(
                data_file,
                dtype='float32',
                mode='r',
                shape=tuple(data_shape)
            )
            for i in range(len(data_array)):
                yield data_array[i], data_array[i]

        # convert to tf.data.Dataset object
        ds = tf.data.Dataset.from_generator(
            data_gen,
            args=[data_file, data_array_shape],
            output_signature=(
                tf.TensorSpec(shape=(height,width,channels), dtype=tf.float32),
                tf.TensorSpec(shape=(height,width,channels), dtype=tf.float32)
            )
        )
        if is_training:
            ds = ds.repeat()
            ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # set sharding to DATA
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)

        return ds


# ------------------------------------------------------------------------------
    

class DomainLearnerFeederFromArray(tf.data.Dataset):
    """
    This class creates an object for feeding data to a model for training. It
    can be used with the model.fit() method in keras. This class takes in a
    numpy array containing the data, and a numpy array containing the labels.
    """
    def __new__(
            cls,
            is_training: bool,
            data_array: np.ndarray,
            label_array: np.ndarray,
            batch_size: int = 32
        ):
        """
        Constructor method for the DomainLearnerFeederFromArray class.

        Parameters
        ----------
        is_training : bool
            A boolean indicating whether the data is for training or validation.
        data_array : numpy.array, optional
            Numpy array containing the data.
            Shape is (n_samples, height, width, channels).
            If the channels dimension is not present, a value of 1 is assumed.
            Values should be in the range 0-255.
        label_array : numpy.array, optional
            Numpy array containing the labels.
            Shape is (n_samples,), with integer-valued labels indicating the
            class of the sample in the range 0 to n_classes-1.
        batch_size : int, optional
            The size of the batch loaded from the generator.
            Default value is 32.

        Returns
        -------
        tf.data.Dataset
            A tf.data.Dataset object containing the data and labels.
        """

        assert len(data_array.shape) == 3 or len(data_array.shape) == 4, \
            "Data array must have 3 or 4 dimensions."
        assert isinstance(data_array, np.ndarray), \
            "Data array must be a numpy array."
        assert isinstance(label_array, np.ndarray), \
            "Label array must be a numpy array."
        assert data_array.shape[0] == label_array.shape[0], \
            "Data and label arrays must have the same number of samples."

        # set constants
        n_samples = data_array.shape[0]
        n_classes = len(np.unique(label_array))

        # convert label array to 1D if necessary
        if len(label_array.shape) > 1:
            label_array = np.reshape(label_array, newshape=(n_samples,))


        # preprocessing
        data_array = data_array.astype(np.float32)/255.0 # normalize to 0-1
        data_array = (data_array - 0.5)/0.5 # normalize to -1 to 1
        if len(data_array.shape) == 3:
            data_array = np.expand_dims(data_array,axis=-1) # add channel dimension
        one_hot_array = np.zeros(shape=(n_samples,n_classes)) # one-hot encode labels
        one_hot_array[np.arange(n_samples),label_array] = 1.0
        one_hot_array = one_hot_array.astype(np.float32)

        # convert to tf.data.Dataset object
        ds = tf.data.Dataset.from_tensor_slices((data_array,(data_array,one_hot_array)))
        if is_training:
            ds = ds.repeat()
            ds = ds.shuffle(buffer_size=n_samples)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # set sharding to DATA
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF
        ds = ds.with_options(options)

        return ds
    

# ------------------------------------------------------------------------------
    

class DomainLearnerFeederFromMemmap(tf.data.Dataset):
    def __new__(
            cls,
            is_training: bool,
            data_file: str,
            label_file: str,
            data_array_shape: tuple,
            label_array_shape: tuple,
            batch_size: int = 32,
            buffer_size: Optional[int] = 10000
        ):
        """
        Constructor method for the ClassifierFeederFromMemmap class.

        Parameters
        ----------
        is_training : bool
            A boolean indicating whether the data is for training or validation.
        data_file : str
            Path to the .npy file containing the data.
            Should be saved as type float32.
        label_file : str
            Path to the .npy file containing the labels.
            Should be saved as type float32.
        data_array_shape : tuple
            Shape of the data array.
            Should have 4 dimensions: (n_samples, height, width, channels).
        label_array_shape : tuple
            Shape of the label array.
            Should have 2 dimensions: (n_samples, n_classes).
        batch_size : int, optional
            The size of the batch loaded from the generator.
            Default value is 32.
        buffer_size : int, optional
            The size of the buffer used for shuffling.
            Default value is 10000.

        Returns
        -------
        tf.data.Dataset
            A tf.data.Dataset object containing the data and labels.
        """

        assert data_array_shape[0] == label_array_shape[0], \
            "Data and label arrays must have the same number of samples."
        
        n_classes = label_array_shape[1]

        height = data_array_shape[1]
        width = data_array_shape[2]
        channels = data_array_shape[3]

        # define generator function
        def data_gen(data_file, label_file, data_shape, label_shape):
            data_array = np.memmap(
                data_file,
                dtype='float32',
                mode='r',
                shape=tuple(data_shape)
            )
            label_array = np.memmap(
                label_file,
                dtype='float32',
                mode='r',
                shape=tuple(label_shape)
            )
            for i in range(len(data_array)):
                yield data_array[i], (data_array[i], label_array[i])

        # convert to tf.data.Dataset object
        ds = tf.data.Dataset.from_generator(
            data_gen,
            args=[data_file, label_file, data_array_shape, label_array_shape],
            output_signature=(
                tf.TensorSpec(shape=(height,width,channels), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(height,width,channels), dtype=tf.float32), 
                    tf.TensorSpec(shape=(n_classes,), dtype=tf.float32)
                )
            )
        )
        if is_training:
            ds = ds.repeat()
            ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # set sharding to DATA
        options = tf.data.Options()
        options.autotune.ram_budget = 10*1024*1024*1024 # 10GB
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF
        ds = ds.with_options(options)

        return ds
    

# ------------------------------------------------------------------------------