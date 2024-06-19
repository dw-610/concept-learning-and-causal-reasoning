"""
This module contains the custom model architectures.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
from typing import Tuple, Union, List, Optional
import tensorflow as tf

from .layers import AWGNLayer, HeightWidthSliceLayer
from .layers import ConvolutionBlock, DeconvolutionBlock
from .layers import EuclideanDistanceLayer
from .layers import SoftGaussSimPredictionLayer
from .layers import SmallResNetBlock, SmallDeResNetBlock
from .layers import ResNetBlock, DeResNetBlock
from .layers import ReparameterizationLayer

from keras import layers, saving

# ------------------------------------------------------------------------------


class ConvEncoder(tf.keras.models.Model):
    """
    This class defines a convolutional autoencoder encoder model.
    """
    def __init__(
            self,
            input_shape: Tuple[int],
            code_size: int,
            number_of_blocks: int = 1,
            filters: Union[int, List[int]] = 16,
            kernel_sizes: Union[int, List[int]] = 3,
            strides: Union[int, List[int]] = 1,
            conv_activation: str = 'relu',
            dense_activation: str = 'linear',
            global_pool_type: Optional[str] = None,
            use_maxpool: Union[bool, List[bool]] = True,
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method for the ConvEncoder class.

        Parameters
        ----------
        input_shape : 3-element tuple        
          Shape of the input tensor, given as a tuple (height, width, channels).
        code_size : int          
            Size of the compressed feature vector.
        number_of_blocks : int, optional
            Number of convolution blocks in the autoencoder. 
            Default value is 1.
        filters : int or list, optional    
            The number of filters used in each convolutional layer.
            If a list is given, its length must be number_of_blocks.
            Default value is 16.
        kernel_sizes : int or list, optional         
            The length/width of the kernels used in the convolutional layers.
            If a list is given, its length must be number_of_blocks. 
            Default value is 3.
        strides : int or list, optional           
            The strides taken in each convolutional layer. Default value is 1.
            If a list is given, its length must be number_of_blocks.
        conv_activation : str, optional  
            String identifier for the convolutional layer activation functions. 
            Default value is 'relu'.
        dense_activation : str, optional
            String identifier for the activation function after the dense layer. 
            Default value is 'linear' (no activation).
        global_pool_type : str, optional
            String identifier for the global pooling type.
            Options are 'max', 'avg', or None. If None, the output of the last
            convolutional layer is just flattened.
            Default value is None.
        use_maxpool : bool or list, optional       
            Whether 2x2 max pooling is used after the convolutional layers. 
            If a list is given, its length must be number_of_blocks.
            Default value is True.
        dropout : float, optional    
            A float between 0 and 1 specifying the rate of dropout used after
            the global pooling/flatten layer.
            If None or 0, no dropout is applied.
            Default value is None.
        """

        super(ConvEncoder, self).__init__(**kwargs)

        # check that list lengths match number of blocks, set flags for lists
        is_filter_list = isinstance(filters, list)
        is_kernel_list = isinstance(kernel_sizes, list)
        is_stride_list = isinstance(strides, list)
        is_maxpool_list = isinstance(use_maxpool, list)
        if is_filter_list:
            assert len(filters) == number_of_blocks, \
                "length of 'filters_list' list should be number_of_blocks"
        if is_kernel_list:
            assert len(kernel_sizes) == number_of_blocks, \
                "length of 'kernel_sizes' list should be number_of_blocks"
        if is_stride_list:
            assert len(strides) == number_of_blocks, \
                "length of 'strides' list should be number_of_blocks"
        if is_maxpool_list:
            assert len(use_maxpool) == number_of_blocks, \
                "length of 'use_maxpool' list should be number_of_blocks"

        self.ip_shape = input_shape
        self.code_size = code_size
        self.number_of_blocks = number_of_blocks
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.pool_type = global_pool_type
        self.use_maxpool = use_maxpool
        self.dropout = dropout

        self.input_layer = layers.InputLayer(input_shape=self.ip_shape)

        self.layers_list = [self.input_layer]

        # add the convolutional block layers
        for i in range(self.number_of_blocks):
            self.layers_list.append(
                ConvolutionBlock(
                    filters=self.filters[i] if is_filter_list else self.filters,
                    kernel_size=self.kernel_sizes[i] if is_kernel_list else \
                        self.kernel_sizes,
                    strides=self.strides[i] if is_stride_list else self.strides,
                    activation=conv_activation,
                    maxpool=self.use_maxpool[i] if is_maxpool_list else \
                        self.use_maxpool
                )
            )

        # add the global pooling layer if specified
        if global_pool_type == 'max':
            self.layers_list.append(layers.GlobalMaxPooling2D())
        elif global_pool_type == 'avg':
            self.layers_list.append(layers.GlobalAveragePooling2D())
        elif global_pool_type is None:
            self.layers_list.append(layers.Flatten())
        else:
            raise ValueError("Got invalid 'global_pool_type'.")

        # if specified, add dropout before the dense layer
        if dropout is not None and dropout > 0:
            self.layers_list.append(layers.Dropout(dropout))

        # add a dense layer to reduce the output to the code size
        self.layers_list.append(
            layers.Dense(
                units=self.code_size, 
                activation=self.dense_activation
            )
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            if layer.name == 'input_layer':
                continue
            else:
                x = layer(x, training=training)
        return x
    
    def summary(self):
        x = tf.keras.Input(shape=(self.ip_shape))
        model = tf.keras.models.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
    def get_config(self):
        config = super(ConvEncoder, self).get_config()
        config.update({
            'input_shape': self.ip_shape,
            'code_size': self.code_size,
            'number_of_blocks': self.number_of_blocks,
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
            'strides': self.strides,
            'conv_activation': self.conv_activation,
            'dense_activation': self.dense_activation,
            'global_pool_type': self.pool_type,
            'use_maxpool': self.use_maxpool,
            'dropout': self.dropout,
        })
        return config

        
# ------------------------------------------------------------------------------


class ConvDecoder(tf.keras.models.Model):
    """
    This class defines a convolutional autoencoder decoder model.
    """
    def __init__(
            self,
            encoder: tf.keras.models.Model,
            output_channels: int,
            code_size: int,
            number_of_blocks: int = 1,
            filters: Union[int, List[int]] = 16,
            kernel_sizes: Union[int, List[int]] = 3,
            strides: Union[int, List[int]] = 1,
            deconv_activation: str = 'relu',
            output_activation: str = 'linear',
            use_unpooling: Union[bool, List[bool]] = True,
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method for the ConvDecoder class.

        Parameters
        ----------
        encoder : tf.keras.models.Model
            Encoder model. This is used to get dimensions for the decoder.
        output_channels : int        
            The number of channels in the output tensor. E.g., 3 for RGB images.
        code_size : int          
            Size of the compressed latent/feature vector.
        number_of_blocks : int, optional
            Number of deconvolution blocks in the autoencoder. 
            Default value is 1.
        filters : int or list, optional    
            The number of filters used in each deconvolutional layer.
            Use the same list as the encoder, will be reversed automatically.
            If a list is given, its length must be number_of_layers.
            Default value is 16.
        kernel_sizes : int or list, optional         
            The length/width of the kernels used in the deconvolutional layers.
            Use the same list as the encoder, will be reversed automatically.
            If a list is given, its length must be number_of_layers. 
            Default value is 3.
        strides : int or list, optional           
            The strides taken in each deconvolutional layer. Default value is 1.
            Use the same list as the encoder, will be reversed automatically.
            If a list is given, its length must be number_of_layers.
        deconv_activation : str, optional 
            String identifier for the deconvolutional layer activation functions. 
            Default value is 'relu'.
        output_activation : str, optional
            String identifier for the output layer activation function. 
            Default value is 'linear' (no activation).
        use_unpooling : bool or list, optional      
            Set this to true if maxpooling is used in the encoder network. 
            Use the same list as the encoder, will be reversed automatically.
            If a list is given, its length must be number_of_blocks.
            Default value is True.
        dropout : float, optional
            A float between 0 and 1 specifying the rate of dropout added after
            the first "flattened" feature layer.
            If None or 0, no dropout is applied.
            Default value is None.
        """

        super(ConvDecoder, self).__init__(**kwargs)

        # check that list lengths match number of layers, set flags for lists
        is_filter_list = isinstance(filters, list)
        is_kernel_list = isinstance(kernel_sizes, list)
        is_stride_list = isinstance(strides, list)
        is_unpool_list = isinstance(use_unpooling, list)
        if is_filter_list:
            assert len(filters) == number_of_blocks, \
                "length of 'filters_list' list should be number_of_blocks"
        if is_kernel_list:
            assert len(kernel_sizes) == number_of_blocks, \
                "length of 'kernel_sizes' list should be number_of_blocks"
        if is_stride_list:
            assert len(strides) == number_of_blocks, \
                "length of 'strides' list should be number_of_blocks"
        if is_unpool_list:
            assert len(use_unpooling) == number_of_blocks, \
                "length of 'use_unpooling' list should be number_of_blocks"
            
        # get the dimensions and dense size from the encoder
        first_height, first_width, \
             first_channels, d_list = self.get_dimensions(encoder)

        # reverse the lists that are given - important that these are NEW lists
        self.filters = list(reversed(filters)) if is_filter_list else filters
        self.kernel_sizes = list(reversed(kernel_sizes)) if is_kernel_list \
            else kernel_sizes
        self.strides = list(reversed(strides)) if is_stride_list else strides
        self.use_unpooling = list(reversed(use_unpooling)) if is_unpool_list \
            else use_unpooling

        # save off parameters
        self.output_channels = output_channels
        self.code_size = code_size
        self.number_of_blocks = number_of_blocks
        self.deconv_activation = deconv_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.d_list = d_list
        self.dense_size = first_channels

        self.ip_shape = (code_size,)

        self.layers_list = []

        self.layers_list.append(
            layers.Dense(
                units=first_height*first_width*first_channels,
                activation=self.deconv_activation
            )
        )

        # if dropout is used, add a dropout layer
        if dropout is not None and dropout > 0:
            self.layers_list.append(layers.Dropout(dropout))
            
        self.layers_list.append(
            layers.Reshape((
                first_height, 
                first_width, 
                first_channels,
            ))
        )

        # add the deconvolutional block layers
        for i in range(number_of_blocks-1):
            if (d_list[i+1]%d_list[i] != 0):
                self.layers_list.append(
                    HeightWidthSliceLayer(1)
                )
            self.layers_list.append(
                DeconvolutionBlock(
                    filters=self.filters[i+1] if is_filter_list else self.filters,
                    kernel_size=self.kernel_sizes[i] if is_kernel_list else self.kernel_sizes,
                    strides=self.strides[i] if is_stride_list else self.strides,
                    activation=deconv_activation,
                    unpool=self.use_unpooling[i] if is_unpool_list else self.use_unpooling
                )
            )

        # add final layer to reduce to input channels
        self.layers_list.append(
            DeconvolutionBlock(
                filters=self.output_channels,
                kernel_size=self.kernel_sizes[-1] if is_kernel_list else self.kernel_sizes,
                strides=self.strides[-1] if is_stride_list else self.strides,
                activation=self.output_activation,
                unpool=self.use_unpooling[-1] if is_unpool_list else self.use_unpooling
            )
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x
    
    def summary(self):
        x = tf.keras.Input(shape=(self.ip_shape))
        model = tf.keras.models.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def get_config(self):
        config = super(ConvDecoder, self).get_config()
        config.update({
            'output_channels': self.output_channels,
            'code_size': self.code_size,
            'number_of_blocks': self.number_of_blocks,
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
            'strides': self.strides,
            'deconv_activation': self.deconv_activation,
            'output_activation': self.output_activation,
            'use_unpooling': self.use_unpooling,
            'dense_size': self.dense_size,
            'd_list': self.d_list,
            'dropout': self.dropout,
            'encoder': saving.serialize_keras_object(self.encoder)
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        submodel_config = config.pop('encoder')
        submodel = saving.deserialize_keras_object(submodel_config)
        return cls(encoder=submodel, **config)

    def get_dimensions(self, encoder):
        d_list = []
        shape = (None,) + encoder.ip_shape
        for layer in encoder.layers:
            if 'input' in layer.name:
                continue
            shape = layer.compute_output_shape(shape)
            if 'convolution' in layer.name:
                d_list.append(shape[1])
                height = shape[1]
                width = shape[2]
                channels = shape[3]
        d_list.reverse()
        return height, width, channels, d_list


# ------------------------------------------------------------------------------


class ResNet18Encoder(tf.keras.models.Model):
    """
    This class defines a ResNet18-based convolutional encoder model.

    See "Deep Residual Learning for Image Recognition" by He et al. 2016 for
    details on the ResNet architecture.
    """
    def __init__(
            self,
            input_shape: Tuple[int],
            code_size: int,
            conv_activation: str = 'relu',
            dense_activation: str = 'linear',
            global_pool_type: str = 'max',
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method for the ResNet18Encoder class.

        Parameters
        ----------
        input_shape : 3-element tuple        
          Shape of the input tensor, given as a tuple (height, width, channels).
        code_size : int          
            Size of the compressed feature vector.
        conv_activation : str, optional  
            String identifier for the convolutional layer activation functions. 
            Default value is 'relu'.
        dense_activation : str, optional
            String identifier for the activation function after the dense layer. 
            Default value is 'linear'.
        global_pool_type : str, optional
            String identifier for the global pooling type.
            Options are 'max' or 'avg'.
            Default value is 'max'.
        dropout : float, optional
            A float between 0 and 1 specifying the rate of dropout used after
            the global pooling/flatten layer.
            If None or 0, no dropout is applied.
            Default value is None.
        """
        super(ResNet18Encoder, self).__init__(**kwargs)

        # save off parameters
        self.ip_shape = input_shape
        self.code_size = code_size
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.global_pool_type = global_pool_type
        self.dropout = dropout

        self.input_layer = layers.InputLayer(input_shape=self.ip_shape)

        self.layers_list = [self.input_layer]

        # layer conv1
        self.layers_list.append(
            ConvolutionBlock(
                filters=64,
                kernel_size=7,
                strides=2,
                activation=conv_activation
            )
        )

        # layer conv2_1
        self.layers_list.append(
            layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        )

        # layers conv2_1, conv2_2
        for i in range(2):
            self.layers_list.append(
                SmallResNetBlock(
                    filters=64,
                    activation=conv_activation,
                    downsample=False
                )
            )

        # layers conv3_1, conv3_2
        for i in range(2):
            self.layers_list.append(
                SmallResNetBlock(
                    filters=128,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # layers conv4_1, conv4_2
        for i in range(2):
            self.layers_list.append(
                SmallResNetBlock(
                    filters=256,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # layers conv5_1, conv5_2
        for i in range(2):
            self.layers_list.append(
                SmallResNetBlock(
                    filters=512,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # add the global pooling layer
        if global_pool_type == 'max':
            self.layers_list.append(layers.GlobalMaxPooling2D())
        elif global_pool_type == 'avg':
            self.layers_list.append(layers.GlobalAveragePooling2D())
        elif global_pool_type is None:
            self.layers_list.append(layers.Flatten())
        else:
            raise ValueError("Got invalid 'global_pool_type'.")

        # if specified, add dropout before the dense layer
        if dropout is not None and dropout > 0:
            self.layers_list.append(layers.Dropout(dropout))

        # add a dense layer to reduce the output to the code size
        self.layers_list.append(
            layers.Dense(
                units=self.code_size, 
                activation=self.dense_activation
            )
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x
    
    def get_config(self):
        config = super(ResNet18Encoder, self).get_config()
        config.update({
            'input_shape': self.ip_shape,
            'code_size': self.code_size,
            'conv_activation': self.conv_activation,
            'dense_activation': self.dense_activation,
            'global_pool_type': self.global_pool_type,
            'dropout': self.dropout,
        })
        return config
    

# ------------------------------------------------------------------------------


class ResNet18Decoder(tf.keras.models.Model):
    """
    This class defines a decoder model meant to mirror the ResNet18 encoder
    model. Built to mirror the ResNet18Encoder model defined above.
    """
    def __init__(
            self,
            encoder: tf.keras.models.Model,
            output_channels: int,
            code_size: int,
            deconv_activation: str = 'relu',
            output_activation: str = 'linear',
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method for the ResNet18Decoder class.

        Parameters
        ----------
        encoder : tf.keras.models.Model
            Encoder model. This is used to get dimensions for the decoder.
        output_channels : int        
            The number of channels in the output tensor.
        code_size : int          
            Size of the compressed feature vector.
        deconv_activation : str, optional 
            String identifier for the deconvolutional layer activation functions. 
            Default value is 'relu'.
        output_activation : str, optional
            String identifier for the output layer activation function. 
            Default value is 'linear'.
        dropout : float, optional
            A float between 0 and 1 specifying the rate of dropout used after
            the first "flattened" feature layer.
            If None or 0, no dropout is applied.
            Default value is None.
        """
        super(ResNet18Decoder, self).__init__(**kwargs)
            
        # get the dimensions to begin reconstruction from the encoder
        first_height, first_width, \
            first_channels, d_list = self.get_dimensions(encoder)

        # save off parameters
        self.output_channels = output_channels
        self.code_size = code_size
        self.deconv_activation = deconv_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.first_height = first_height
        self.first_width = first_width
        self.first_channels = first_channels

        self.layers_list = []

        # get to the shape required by the first convolutional layer
        self.layers_list.append(
            layers.Dense(
                units=first_height*first_width*first_channels,
                activation=deconv_activation
            )
        )

        # if dropout is used, add a dropout layer
        if dropout is not None and dropout > 0:
            self.layers_list.append(layers.Dropout(dropout))

        self.layers_list.append(
            layers.Reshape((
                first_height, 
                first_width, 
                first_channels,
            ))
        )

        # add the de-resnet blocks
        for i in range(2):
            self.layers_list.append(
                SmallDeResNetBlock(
                    filters=512,
                    activation=deconv_activation,
                    upsample=True if i==1 else False
                )
            )
        if (d_list[-5]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(2):
            self.layers_list.append(
                SmallDeResNetBlock(
                    filters=256,
                    activation=deconv_activation,
                    upsample=True if i==1 else False
                )
            )
        if (d_list[-7]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(2):
            self.layers_list.append(
                SmallDeResNetBlock(
                    filters=128,
                    activation=deconv_activation,
                    upsample=True if i==1 else False
                )
            )
        if (d_list[-9]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(2):
            self.layers_list.append(
                SmallDeResNetBlock(
                    filters=64,
                    activation=deconv_activation,
                    upsample=False
                )
            )

        # add a deconv layer to "undo" maxpooling2d layer
        self.layers_list.append(
            layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=2,
                activation=deconv_activation,
                padding='same'
            )
        )
        if (d_list[-12]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        # add a deconv block to undo first convoltuion block
        self.layers_list.append(
            DeconvolutionBlock(
                filters=self.output_channels,
                kernel_size=7,
                strides=2,
                activation=output_activation,
                unpool=False,
            )
        )
        if (d_list[-13]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(ResNet18Decoder, self).get_config()
        config.update({
            'output_channels': self.output_channels,
            'code_size': self.code_size,
            'deconv_activation': self.deconv_activation,
            'output_activation': self.output_activation,
            'dropout': self.dropout,
        })
        return config

    def get_dimensions(self, encoder):
        shape = encoder.input_layer.get_input_shape_at(0)
        d_list = []
        for layer in encoder.layers:
            shape = layer.compute_output_shape(shape)
            name = layer.name
            if ('block_7' in name):
                height = shape[1]
                width = shape[2]
                channels = shape[3]
            d_list.append(shape[1])
        return height, width, channels, d_list


# ------------------------------------------------------------------------------


class ResNet34Encoder(tf.keras.models.Model):
    """
    This class defines a ResNet34-based convolutional encoder model.

    See "Deep Residual Learning for Image Recognition" by He et al. 2016 for
    details on the ResNet architecture.
    """
    def __init__(
            self,
            input_shape: Tuple[int],
            code_size: int,
            conv_activation: str = 'relu',
            dense_activation: str = 'linear',
            global_pool_type: str = 'max',
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method for the ResNet34Encoder class.

        Parameters
        ----------
        input_shape : 3-element tuple        
          Shape of the input tensor, given as a tuple (height, width, channels).
        code_size : int          
            Size of the compressed feature vector.
        conv_activation : str, optional  
            String identifier for the convolutional layer activation functions. 
            Default value is 'relu'.
        dense_activation : str, optional
            String identifier for the activation function after the dense layer. 
            Default value is 'linear'.
        global_pool_type : str, optional
            String identifier for the global pooling type.
            Options are 'max' or 'avg'.
            Default value is 'max'.
        dropout : float, optional
            A float between 0 and 1 specifying the rate of dropout used after
            the global pooling/flatten layer.
            If None or 0, no dropout is applied.
            Default value is None.
        """
        super(ResNet34Encoder, self).__init__(**kwargs)

        # save off parameters
        self.ip_shape = input_shape
        self.code_size = code_size
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.dropout = dropout

        self.input_layer = layers.InputLayer(input_shape=self.ip_shape)

        self.layers_list = [self.input_layer]

        # layer conv1
        self.layers_list.append(
            ConvolutionBlock(
                filters=64,
                kernel_size=7,
                strides=2,
                activation=conv_activation
            )
        )

        # layer conv2_1
        self.layers_list.append(
            layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        )

        # layers conv2_1, conv2_2, conv2_3
        for i in range(3):
            self.layers_list.append(
                SmallResNetBlock(
                    filters=64,
                    activation=conv_activation,
                    downsample=False
                )
            )

        # layers conv3_1, conv3_2, conv3_3, conv3_4
        for i in range(4):
            self.layers_list.append(
                SmallResNetBlock(
                    filters=128,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # layers conv4_1, conv4_2, conv4_3, conv4_4, conv4_5, conv4_6
        for i in range(6):
            self.layers_list.append(
                SmallResNetBlock(
                    filters=256,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # layers conv5_1, conv5_2, conv5_3
        for i in range(3):
            self.layers_list.append(
                SmallResNetBlock(
                    filters=512,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # add the global pooling layer
        if global_pool_type == 'max':
            self.layers_list.append(layers.GlobalMaxPooling2D())
        elif global_pool_type == 'avg':
            self.layers_list.append(layers.GlobalAveragePooling2D())
        elif global_pool_type is None:
            self.layers_list.append(layers.Flatten())
        else:
            raise ValueError("Got invalid 'global_pool_type'.")

        # if specified, add dropout before the dense layer
        if dropout is not None and dropout > 0:
            self.layers_list.append(layers.Dropout(dropout))

        # add a dense layer to reduce the output to the code size
        self.layers_list.append(
            layers.Dense(units=self.code_size, activation=self.dense_activation)
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x
    
    def get_config(self):
        config = super(ResNet34Encoder, self).get_config()
        config.update({
            'input_shape': self.ip_shape,
            'code_size': self.code_size,
            'conv_activation': self.conv_activation,
            'dense_activation': self.dense_activation,
            'dropout': self.dropout,
        })
        return config


# ------------------------------------------------------------------------------


class ResNet34Decoder(tf.keras.models.Model):
    """
    This class defines a decoder model meant to mirror the ResNet34 model.
    Built to mirror the ResNet34Encoder model defined above.
    """
    def __init__(
            self,
            encoder: tf.keras.models.Model,
            output_channels: int,
            code_size: int,
            deconv_activation: str = 'relu',
            output_activation: str = 'linear',
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method for the ResNet34Decoder class.

        Parameters
        ----------
        encoder : tf.keras.models.Model
            Encoder model - used to get dimensions for the decoder.
        output_channels : int        
            The number of channels in the output tensor.
        code_size : int          
            Size of the compressed feature vector.
        deconv_activation : str, optional 
            String identifier for the deconvolutional layer activation functions. 
            Default value is 'relu'.
        output_activation : str, optional
            String identifier for the output layer activation function. 
            Default value is 'linear'.
        dropout : float, optional
            A float between 0 and 1 specifying the rate of dropout used after
            the first "flattened" feature layer.
            If None or 0, no dropout is applied.
            Default value is None.
        """
        super(ResNet34Decoder, self).__init__(**kwargs)
            
        # get the dimensions to begin reconstruction from the encoder
        first_height, first_width, \
            first_channels, d_list = self.get_dimensions(encoder)

        # save off parameters
        self.output_channels = output_channels
        self.code_size = code_size
        self.deconv_activation = deconv_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.first_height = first_height
        self.first_width = first_width
        self.first_channels = first_channels

        self.layers_list = []

        # get to the shape required by the first convolutional layer
        self.layers_list.append(
            layers.Dense(
                units=first_height*first_width*first_channels,
                activation=deconv_activation
            )
        )

        # if dropout is used, add a dropout layer
        if dropout is not None and dropout > 0:
            self.layers_list.append(layers.Dropout(dropout))

        self.layers_list.append(
            layers.Reshape((
                first_height, 
                first_width, 
                first_channels,
            ))
        )

        # add the de-resnet blocks
        for i in range(3):
            self.layers_list.append(
                SmallDeResNetBlock(
                    filters=512,
                    activation=deconv_activation,
                    upsample=True if i==2 else False
                )
            )
        if (d_list[-6]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(6):
            self.layers_list.append(
                SmallDeResNetBlock(
                    filters=256,
                    activation=deconv_activation,
                    upsample=True if i==5 else False
                )
            )
        if (d_list[-12]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(4):
            self.layers_list.append(
                SmallDeResNetBlock(
                    filters=128,
                    activation=deconv_activation,
                    upsample=True if i==3 else False
                )
            )
        if (d_list[-16]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(3):
            self.layers_list.append(
                SmallDeResNetBlock(
                    filters=64,
                    activation=deconv_activation,
                    upsample=False
                )
            )

        # add a deconv layer to "undo" maxpooling2d layer
        self.layers_list.append(
            layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=2,
                activation=deconv_activation,
                padding='same'
            )
        )
        if (d_list[-20]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        # add a deconv block to undo first convoltuion block
        self.layers_list.append(
            DeconvolutionBlock(
                filters=self.output_channels,
                kernel_size=7,
                strides=2,
                activation=output_activation,
                unpool=False,
            )
        )
        if (d_list[-21]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(ResNet34Decoder, self).get_config()
        config.update({
            'output_channels': self.output_channels,
            'code_size': self.code_size,
            'deconv_activation': self.deconv_activation,
            'output_activation': self.output_activation,
            'dropout': self.dropout,
        })
        return config

    def get_dimensions(self, encoder):
        """
        This method gets the dimensions of the spatial layer to reconstruct.
        """
        shape = encoder.input_layer.get_input_shape_at(0)
        d_list = []
        for layer in encoder.layers:
            shape = layer.compute_output_shape(shape)
            name = layer.name
            if ('block_15' in name):
                height = shape[1]
                width = shape[2]
                channels = shape[3]
            d_list.append(shape[1])
        return height, width, channels, d_list


# ------------------------------------------------------------------------------


class ResNet50Encoder(tf.keras.models.Model):
    """
    This class defines a ResNet50-based convolutional encoder model.

    See "Deep Residual Learning for Image Recognition" by He et al. 2016 for
    details on the ResNet architecture.
    """
    def __init__(
            self,
            input_shape: Tuple[int],
            code_size: int,
            conv_activation: str = 'relu',
            dense_activation: str = 'linear',
            global_pool_type: str = 'max',
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method for the ResNet50Encoder class.

        Parameters
        ----------
        input_shape : 3-element tuple        
          Shape of the input tensor, given as a tuple (height, width, channels).
        code_size : int          
            Size of the compressed feature vector.
        conv_activation : str, optional  
            String identifier for the convolutional layer activation functions. 
            Default value is 'relu'.
        dense_activation : str, optional
            String identifier for the activation function after the dense layer. 
            Default value is 'linear'.
        global_pool_type : str, optional
            String identifier for the global pooling type.
            Options are 'max' or 'avg'.
            Default value is 'max'.
        dropout : float, optional
            A float between 0 and 1 specifying the rate of dropout used after
            the global pooling/flatten layer.
            If None or 0, no dropout is applied.
            Default value is None.
        """
        super(ResNet50Encoder, self).__init__(**kwargs)

        # save off parameters
        self.ip_shape = input_shape
        self.code_size = code_size
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.dropout = dropout

        self.input_layer = layers.InputLayer(input_shape=self.ip_shape)

        self.layers_list = [self.input_layer]

        # layer conv1
        self.layers_list.append(
            ConvolutionBlock(
                filters=64,
                kernel_size=7,
                strides=2,
                activation=conv_activation
            )
        )
        

        # layer conv2_1
        self.layers_list.append(
            layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        )

        # layers conv2_1-3
        for i in range(3):
            self.layers_list.append(
                ResNetBlock(
                    filters=256,
                    activation=conv_activation,
                    match_filters=True if i==0 else False
                )
            )

        # layers conv3_1-4
        for i in range(4):
            self.layers_list.append(
                ResNetBlock(
                    filters=512,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # layers conv4_1-6
        for i in range(6):
            self.layers_list.append(
                ResNetBlock(
                    filters=1024,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # layers conv5_1-3
        for i in range(3):
            self.layers_list.append(
                ResNetBlock(
                    filters=2048,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # add the global pooling layer
        if global_pool_type == 'max':
            self.layers_list.append(layers.GlobalMaxPooling2D())
        elif global_pool_type == 'avg':
            self.layers_list.append(layers.GlobalAveragePooling2D())
        elif global_pool_type is None:
            self.layers_list.append(layers.Flatten())
        else:
            raise ValueError("Got invalid 'global_pool_type'.")

        # if specified, add dropout before the dense layer
        if dropout is not None and dropout > 0:
            self.layers_list.append(layers.Dropout(dropout))

        # add a dense layer to reduce the output to the code size
        self.layers_list.append(
            layers.Dense(units=self.code_size, activation=self.dense_activation)
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x
    
    def get_config(self):
        config = super(ResNet50Encoder, self).get_config()
        config.update({
            'input_shape': self.ip_shape,
            'code_size': self.code_size,
            'conv_activation': self.conv_activation,
            'dense_activation': self.dense_activation,
            'dropout': self.dropout,
        })
        return config
    

# ------------------------------------------------------------------------------


class ResNet50Decoder(tf.keras.models.Model):
    """
    This class defines a decoder model meant to mirror the ResNet50 model.
    Built to mirror the ResNet50Encoder model defined above.

    See "Deep Residual Learning for Image Recognition" by He et al. 2016 for
    details on the ResNet architecture.
    """
    def __init__(
            self,
            encoder: tf.keras.models.Model,
            output_channels: int,
            code_size: int,
            deconv_activation: str = 'relu',
            output_activation: str = 'linear',
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method.

        Parameters
        ----------
        encoder : tf.keras.models.Model
            Encoder model - used to get dimensions for the decoder.
        output_channels : int        
            The number of channels in the output tensor.
        code_size : int          
            Size of the compressed feature vector.
        deconv_activation : str, optional 
            String identifier for the deconvolutional layer activation functions. 
            Default value is 'relu'.
        output_activation : str, optional
            String identifier for the output layer activation function. 
            Default value is 'linear'.
        dropout : float, optional
            A float between 0 and 1 specifying the rate of dropout used after
            the first "flattened" feature layer.
            If None or 0, no dropout is applied.
            Default value is None.
        """

        super(ResNet50Decoder, self).__init__(**kwargs)
            
        # get the dimensions to begin reconstruction from the encoder
        first_height, first_width, \
            first_channels, d_list = self.get_dimensions(encoder)

        # save off parameters
        self.output_channels = output_channels
        self.code_size = code_size
        self.deconv_activation = deconv_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.first_height = first_height
        self.first_width = first_width
        self.first_channels = first_channels

        self.layers_list = []

        # get to the shape required by the first convolutional layer
        self.layers_list.append(
            layers.Dense(
                units=first_height*first_width*first_channels,
                activation=deconv_activation
            )
        )

        # if dropout is used, add a dropout layer
        if dropout is not None and dropout > 0:
            self.layers_list.append(layers.Dropout(dropout))

        self.layers_list.append(
            layers.Reshape((
                first_height, 
                first_width, 
                first_channels,
            ))
        )

        # add the de-resnet blocks
        for i in range(3):
            self.layers_list.append(
                DeResNetBlock(
                    filters=2048,
                    activation=deconv_activation,
                    upsample=True if i==2 else False
                )
            )
        if (d_list[-6]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(6):
            self.layers_list.append(
                DeResNetBlock(
                    filters=1024,
                    activation=deconv_activation,
                    upsample=True if i==5 else False
                )
            )
        if (d_list[-12]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(4):
            self.layers_list.append(
                DeResNetBlock(
                    filters=512,
                    activation=deconv_activation,
                    upsample=True if i==3 else False
                )
            )
        if (d_list[-16]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        for i in range(3):
            self.layers_list.append(
                DeResNetBlock(
                    filters=256,
                    activation=deconv_activation,
                    upsample=False
                )
            )

        # add a deconv layer to "undo" maxpooling2d layer
        self.layers_list.append(
            layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=2,
                activation=deconv_activation,
                padding='same'
            )
        )
        if (d_list[-20]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

        # add a deconv block to undo first convoltuion block
        self.layers_list.append(
            DeconvolutionBlock(
                filters=self.output_channels,
                kernel_size=7,
                strides=2,
                activation=output_activation,
                unpool=False
            )
        )
        if (d_list[-21]%2 != 0):
            self.layers_list.append(
                HeightWidthSliceLayer(1)
            )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(ResNet50Decoder, self).get_config()
        config.update({
            'output_channels': self.output_channels,
            'code_size': self.code_size,
            'deconv_activation': self.deconv_activation,
            'output_activation': self.output_activation,
            'dropout': self.dropout,
        })
        return config

    def get_dimensions(self, encoder):
        """
        This method gets the dimensions of the spatial layer to reconstruct.
        """
        shape = encoder.input_layer.get_input_shape_at(0)
        d_list = []
        for layer in encoder.layers:
            shape = layer.compute_output_shape(shape)
            name = layer.name
            if ('block_15' in name):
                height = shape[1]
                width = shape[2]
                channels = shape[3]
            d_list.append(shape[1])
        return height, width, channels, d_list


# ------------------------------------------------------------------------------


class ResNet101Encoder(tf.keras.models.Model):
    """
    This class defines a ResNet101-based convolutional encoder model.

    See "Deep Residual Learning for Image Recognition" by He et al. 2016 for
    details on the ResNet architecture.
    """
    def __init__(
            self,
            input_shape: Tuple[int],
            code_size: int,
            conv_activation: str = 'relu',
            dense_activation: str = 'linear',
            global_pool_type: str = 'max',
            dropout: Optional[float] = None,
            **kwargs
        ):
        """
        Constructor method.

        Parameters
        ----------
        input_shape : 3-element tuple        
          Shape of the input tensor, given as a tuple (height, width, channels).
        code_size : int          
            Size of the compressed feature vector.
        conv_activation : str, optional  
            String identifier for the convolutional layer activation functions. 
            Default value is 'relu'.
        dense_activation : str, optional
            String identifier for the activation function after the dense layer. 
            Default value is 'linear'.
        global_pool_type : str, optional
            String identifier for the global pooling type.
            Options are 'max' or 'avg'.
            Default value is 'max'.
        dropout : float, optional
            A float between 0 and 1 specifying the rate of dropout.
            If None or 0, no dropout is applied.
            Default value is None.
        """

        super(ResNet101Encoder, self).__init__(**kwargs)

        # save off parameters
        self.ip_shape = input_shape
        self.code_size = code_size
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.dropout = dropout

        self.input_layer = layers.InputLayer(input_shape=self.ip_shape)

        self.layers_list = [self.input_layer]

        # layer conv1
        self.layers_list.append(
            ConvolutionBlock(
                filters=64,
                kernel_size=7,
                strides=2,
                activation=conv_activation
            )
        )

        # layer conv2_1
        self.layers_list.append(
            layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        )

        # layers conv2_1-3
        for i in range(3):
            self.layers_list.append(
                ResNetBlock(
                    filters=256,
                    activation=conv_activation,
                    match_filters=True if i==0 else False
                )
            )

        # layers conv3_1-4
        for i in range(4):
            self.layers_list.append(
                ResNetBlock(
                    filters=512,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # layers conv4_1-23
        for i in range(23):
            self.layers_list.append(
                ResNetBlock(
                    filters=1024,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # layers conv5_1-3
        for i in range(3):
            self.layers_list.append(
                ResNetBlock(
                    filters=2048,
                    activation=conv_activation,
                    downsample=True if i==0 else False
                )
            )

        # add the global pooling layer
        if global_pool_type == 'max':
            self.layers_list.append(layers.GlobalMaxPooling2D())
        elif global_pool_type == 'avg':
            self.layers_list.append(layers.GlobalAveragePooling2D())
        elif global_pool_type is None:
            self.layers_list.append(layers.Flatten())
        else:
            raise ValueError("Got invalid 'global_pool_type'.")

        # add a dense layer to reduce the output to the code size
        self.layers_list.append(
            layers.Dense(units=self.code_size, activation=self.dense_activation)
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x
    
    def get_config(self):
        config = super(ResNet101Encoder, self).get_config()
        config.update({
            'input_shape': self.ip_shape,
            'code_size': self.code_size,
            'conv_activation': self.conv_activation,
            'dense_activation': self.dense_activation,
            'dropout': self.dropout
        })
        return config
    

# ------------------------------------------------------------------------------


class Classifier(tf.keras.models.Model):
    """
    This class defines a simple classifier model, based on a predefined encoder
    model.
    """
    def __init__(
            self,
            num_classes: int,
            encoder: tf.keras.models.Model,
            output_activation: str = 'linear',
            use_awgn: bool = False,
            awgn_variance: float = 1.0,
            **kwargs
        ):
        """
        Constructor method for the Classifier class.

        Parameters
        ----------
        num_classes : int
            The number of classes in the classification problem.
        encoder : tf.keras.models.Model
            The encoder model to use for feature extraction.
        output_activation : str, optional
            The activation function to use for the output layer.
            Default value is 'linear'.
        use_awgn : bool, optional
            Whether to use additive white Gaussian noise (AWGN) in the model.
            Default value is False.
        awgn_variance : float, optional
            The variance of the AWGN to use, if enabled.
            Default value is 1.0.
        """
        super(Classifier, self).__init__()

        self.ip_shape = encoder.ip_shape

        self.num_classes = num_classes
        self.op_act = output_activation
        self.use_awgn = use_awgn
        self.awgn_variance = awgn_variance

        if use_awgn:
            self.awgn_layer = AWGNLayer(awgn_variance)
        else:
            self.awgn_layer = None

        self.encoder = encoder

        self.dense_layer = layers.Dense(
            num_classes, 
            activation=output_activation,
            use_bias=False
        )
        
    def compile(
            self,
            loss: Optional[Union[str, tf.keras.losses.Loss]] = None,
            metric_matrix: Optional[np.array] = None,
            wasserstein_lam: Optional[float] = 1.0,
            wasserstein_p: Optional[float] = 1.0,
            **kwargs
        ):
        """
        Function for compiling the Classifier object. Accepts the same 
        parameters as the tf.keras.models.Model.compile function, with the 
        addition of the metric_matrix and wasserstein_lam parameters.

        Parameters
        ----------
        loss : str or tf.keras.losses.Loss, optional
            The loss function to use for training.
            Currently, the only str accepted is 'wasserstein'.
            For traditional losses, use the tf.keras.losses.Loss objects.
            Default value is None.
        metric_matrix : numpy.array, optional
            The metric matrix to use for the Wasserstein loss.
            Required if using the Wasserstein loss.
            Default value is None.
        wasserstein_lam : float, optional
            The lambda value to use for the Wasserstein loss.
            Default value is 1.0.
        wasserstein_p : float, optional
            The p value to use for the Wasserstein loss.
            Default value is 1.0.
        """
        if loss == 'wasserstein':
            if metric_matrix is not None:
                self.metric_matrix = tf.Variable(metric_matrix, trainable=False)
            else:
                raise ValueError(
                    "Must provide a metric matrix for the Wasserstein loss."
                )
            self.wasserstein_lam = wasserstein_lam
            self.wasserstein_p = wasserstein_p
        super(Classifier, self).compile(
            **kwargs,
            loss=loss if type(loss) is not str else None
        )

    def call(self, inputs, training=False):
        if self.use_awgn:
            x = self.awgn_layer(inputs, training=training)
            x = self.encoder(x, training=training)
        else:
            x = self.encoder(inputs, training=training)
        x = self.dense_layer(x, training=training)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            'num_classes': self.num_classes,
            'encoder': saving.serialize_keras_object(self.encoder),
            'output_activation': self.op_act,
            'use_awgn': self.use_awgn,
            'awgn_variance': self.awgn_variance,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        submodel_config = config.pop('encoder')
        submodel = saving.deserialize_keras_object(submodel_config)
        return cls(encoder=submodel, **config)
    
    def summary(self):
        x = tf.keras.Input(shape=(self.ip_shape))
        model = tf.keras.models.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


# ------------------------------------------------------------------------------


class Autoencoder(tf.keras.models.Model):
    """
    This class defines a general end-to-end autoencoder model by taking in an 
    already-defined encoder and decoder.
    """
    def __init__(
            self,
            encoder: tf.keras.models.Model,
            decoder: tf.keras.models.Model,
            use_awgn: bool = False,
            awgn_variance: float = 1.0,
            **kwargs
        ):
        """
        Constructor method.

        Parameters
        ----------
        encoder : tf.keras.models.Model
            Encoder model.
        decoder : tf.keras.models.Model
            Decoder model.
        use_awgn : bool, optional
            Whether to use additive white Gaussian noise.
            Default value is False.
        awgn_variance : float, optional
            The variance of the Gaussian noise, if added.
            Default value is 1.0.
        """
        super(Autoencoder, self).__init__(**kwargs)

        self.use_awgn = use_awgn
        self.awgn_variance = awgn_variance

        if self.use_awgn:
            self.awgn_layer = AWGNLayer(self.awgn_variance)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False): 
        if self.use_awgn:
            x = self.awgn_layer(inputs, training=training)
        else:
            x = inputs
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)

        return x
    
    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({
            'encoder': self.encoder,
            'decoder': self.decoder,
            'use_awgn': self.use_awgn,
            'awgn_variance': self.awgn_variance
        })
        return config

    
# ------------------------------------------------------------------------------


class DomainLearnerModel(tf.keras.models.Model):
    """
    This class creates a model for use in the DomainLearning class. There are
    three primary components:
    1) An encoder model, which maps raw data to conceptual space dimensions.
    2) A decoder model, attempting to reconstruct the data from the features.
    3) A minimum-distance decoder, attempting to get predict the property.

    See D. Wheeler, B. Natarajan, "Autoencoder-Based Domain Learning for
    Semantic Communication with Conceptual Spaces," 2024 for more details. 
    """

    def __init__(
            self,
            encoder: tf.keras.models.Model,
            decoder: tf.keras.models.Model,
            initial_prototypes: np.array,
            latent_dimension: int,
            autoencoder_type: str = 'standard',
            distance: str = 'euclidean',
            similarity: str = 'gaussian',
            similarity_c: float = 1.0,
            use_awgn: bool = False,
            awgn_variance: float = 1.0,
            **kwargs
        ):
        """
        Constructor method.

        Parameters
        ----------
        encoder : tf.keras.models.Model
            The encoder portion of the network.
        decoder : tf.keras.models.Model
            The autoencoder decoder portion of the network.
        intial_protos : numpy.array
            The initial prototype points of the properties in the domain.
            The number of rows should be equal to the number of prototypes.
            The number of columns should be equal to the number of dimensions.
        latent_dimension : int
            The number of dimensions in the latent space.
        autoencoder_type : str, optional
            String identifier for the type of autoencoder to be used.
            Options are 'standard' or 'variational'.
            Default value is 'standard'.
        distance : str, optional
            String identifier for the distance function to be used.
            Currently, only option is 'euclidean'.
            Default value is 'euclidean'
        similarity : str, optional
            String identifier for the similarity function to be used.
            Currently, only option is 'gaussian'.
            Default value is 'gaussian'
        similarity_c : float, optional
            Hyperparameter controlling the "stretch" of the similarity function.
            Default value is 1.0.
        use_awgn : bool, optional
            Whether to use additive white Gaussian noise.
            Default value is False.
        awgn_variance : float, optional
            The variance of the Gaussian noise, if added.
            Default value is 1.0.
        """
        super(DomainLearnerModel, self).__init__()

        self.use_awgn = use_awgn
        self.awgn_variance = awgn_variance
        self.dist = distance
        self.sim = similarity
        self.sim_c = similarity_c
        self.latent_dim = latent_dimension
        self.ae_type = autoencoder_type

        self.ip_shape = encoder.ip_shape

        if self.use_awgn:
            self.awgn_layer = AWGNLayer(self.awgn_variance)

        self.encoder = encoder
        if self.ae_type == 'variational':
            self.reparam_layer = ReparameterizationLayer(latent_dimension)
        elif self.ae_type == 'standard':
            pass
        else:
            raise ValueError('Got invalid autoencoder_type.')
        self.decoder = decoder

        self.protos = tf.Variable(
            initial_prototypes,
            trainable=False,
            dtype=tf.float32
        )
        
        if distance == 'euclidean':
            self.distance_layer = EuclideanDistanceLayer(self.protos)
        else: 
            raise ValueError('Got invalid distance function.')

        if similarity == 'gaussian':
            self.soft_sim_layer = SoftGaussSimPredictionLayer(self.sim_c)
        else:
            raise ValueError('Got invalid similarity function.')

        self.training_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='training_accuracy'
        )

        self.validation_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='validation_accuracy'
        )

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.cat_cross = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            axis=-1
        )

    def compile(
            self,
            loss: Optional[str] = None,
            alpha: float = 1.0,
            beta: float = 1.0,
            lam: float = 0.001,
            metric_matrix: Optional[np.array] = None,
            wasserstein_lam: float = 1.0,
            wasserstein_p: float = 1.0,
            scaled_prior: bool = False,
            **kwargs
        ):
        """
        Function for compiling the model. This is where the loss functions and
        optimizer are defined.

        Parameters
        ----------
        alpha : float, optional
            Hyperparameter controlling the scale of the reconstruction loss.
            Default value is 1.0.
        beta : float, optional
            Hyperparameter controlling the scale of the classification loss.
            Default value is 1.0.
        lam : float, optional
            The weight allocated to the regularization term.
            If autoencoder_type is 'variational', this is the weight for the
            KL divergence term. If autoencoder_type is 'standard', this is the
            weight for the custom distance regularization term.
            Default value is 0.001.
        other arguments
            Other arguments to pass to tf.keras.models.Model.compile().
        """
        self.alpha = alpha
        self.beta_val = beta
        self.beta = tf.Variable(beta, trainable=False)
        self.lam = lam
        self.scaled_prior = scaled_prior

        if loss == 'wasserstein':
            if metric_matrix is not None:
                self.metric_matrix = tf.Variable(metric_matrix, trainable=False)
                self.M_fixed = True
            else:
                self.metric_matrix = tf.Variable(
                    tf.zeros(
                        (self.protos.shape[0], self.protos.shape[0])
                    ),
                    trainable=False
                )
                self.M_fixed = False
            self.wasserstein_lam = wasserstein_lam
            self.wasserstein_p = wasserstein_p
        else:
            self.metric_matrix = None
            self.wasserstein_lam = None
            self.wasserstein_p = None
            self.lam = lam

        super(DomainLearnerModel, self).compile(**kwargs)
 
    def call(self, x, training=False):
        """
        Function for defining the logic of the model.

        Parameters
        ----------
        x : tensorflow.Tensor
            The input data. Takes in an image tensor.

        Returns
        -------
        reconstructed : tensorflow.Tensor
            The reconstructed input.
        properties : tensorflow.Tensor
            The predicted properties.
        distances : tensorflow.Tensor
            The distances to the prototypes.
        """

        # get the features from the encoder
        if self.use_awgn:
            noisy = self.awgn_layer(x, training=training)
            features = self.encoder(noisy, training=training)
        else:
            features = self.encoder(x, training=training)

        # get the reconstructed input with the autoencoder decoder
        if self.ae_type == 'variational':
            outputs = self.reparam_layer(features)
            if isinstance(outputs, list):
                log_stds, mus, features = outputs
            else:
                features = outputs
            reconstructed = self.decoder(features)
        elif self.ae_type == 'standard':
            reconstructed = self.decoder(features)
        else:
            raise ValueError('Got invalid autoencoder_type')

        # get the distances and similarity-based predictions
        distances = self.distance_layer(features)
        properties = self.soft_sim_layer(distances)

        if self.ae_type == 'variational':
            if isinstance(outputs, list):
                return [reconstructed, properties, distances, log_stds, mus]
            else:
                return [reconstructed, properties, distances]
        elif self.ae_type == 'standard':
            return [reconstructed, properties, distances]
        else:
            raise ValueError('Got invalid autoencoder_type')
        
    def summary(self):
        x = tf.keras.Input(shape=(self.ip_shape))
        model = tf.keras.models.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            if self.ae_type == 'variational':
                pred_ims, props, _, log_stds, mus = self(x, training=True)
                r_loss = tf.reduce_mean(
                    self.reconstruction_loss(y[0], pred_ims),
                    axis=-1
                )
                c_loss = self.cat_cross(y[1], props)
                kl_loss = tf.reduce_mean(
                    self.kl_divergence_loss(log_stds, mus),
                    axis=-1
                )
                loss = -self.alpha*r_loss + self.beta*c_loss + self.lam*kl_loss
            elif self.ae_type == 'standard':
                pred_ims, props, dists = self(x, training=True)
                r_loss = self.mse_loss(y[0], pred_ims)
                c_loss = self.cat_cross(y[1], props)
                d_reg = self.distance_regularization(dists, y[1])
                loss = self.alpha*r_loss + self.beta*c_loss + self.lam*d_reg
            else:
                raise ValueError('Got invalid autoencoder_type')

        # compute gradients and apply them
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.training_accuracy.update_state(y[1], props)

        if self.ae_type == 'variational':
            return {
                "loss": loss,
                "wl_r": -self.alpha*r_loss,
                "wl_c": self.beta*c_loss,
                "wl_kl": self.lam*kl_loss,
                "l_r": -r_loss,
                "l_c": c_loss,
                "l_kl": kl_loss,
                "accuracy": self.training_accuracy.result()
            }
        elif self.ae_type == 'standard':
            return {
                "loss": loss,
                "wl_r": self.alpha*r_loss,
                "wl_c": self.beta*c_loss,
                "wl_d": self.lam*d_reg,
                "l_r": r_loss,
                "l_c": c_loss,
                "l_d": d_reg,
                "accuracy": self.training_accuracy.result()
            }
    
    def test_step(self, data):
        x, y = data
        if self.ae_type == 'variational':
            pred_ims, props, _, log_stds, mus = self(x, training=False)
            r_loss = tf.reduce_mean(
                self.reconstruction_loss(y[0], pred_ims),
                axis=-1
            )
            c_loss = self.cat_cross(y[1], props)
            kl_loss = tf.reduce_mean(
                self.kl_divergence_loss(log_stds, mus),
                axis=-1
            )
            loss = -self.alpha*r_loss + self.beta*c_loss + self.lam*kl_loss
        elif self.ae_type == 'standard':
            pred_ims, props, dists = self(x, training=False)
            r_loss = self.mse_loss(y[0], pred_ims)
            c_loss = self.cat_cross(y[1], props)
            d_reg = self.distance_regularization(dists, y[1])
            loss = self.alpha*r_loss + self.beta*c_loss + self.lam*d_reg
        else:
            raise ValueError('Got invalid autoencoder_type')

        # Update metrics
        self.validation_accuracy.update_state(y[1], props)

        if self.ae_type == 'variational':
            return {
                "loss": loss,
                "l_r": -r_loss,
                "l_c": c_loss,
                "l_kl": kl_loss,
                "accuracy": self.validation_accuracy.result()
            }
        elif self.ae_type == 'standard':
            return {
                "loss": loss,
                "l_r": r_loss,
                "l_c": c_loss,
                "l_d": d_reg,
                "accuracy": self.validation_accuracy.result()
            }
        else:
            raise ValueError('Got invalid autoencoder_type')
    
    def reconstruction_loss(self, y_true, y_pred):
        """
        Function for the image reconstruction loss when using a variational
        autoencoder.

        Basic assumption is that the output variables are conditionally 
        independent and Gaussian dsitributed given the latent vector. 
        Additionally, it is assumed that the variance is constant across the
        output variables. This variance is captured by the regularization
        weight assigned to the KL divergence term.

        Parameters
        ----------
        y_true : tensorflow.Tensor
            The true data (labels).
        y_pred : tensorflow.Tensor
            The reconstructed data. These are technically the means of the
            conditional Gaussian distributions.

        Returns
        -------
        loss : float
            The loss for each data point. Will be a tensor of shape (None,).
        """
        loss = -tf.reduce_sum(tf.square(y_true - y_pred), axis=[1,2,3])
        return loss

    def kl_divergence_loss(self, predicted_sigma, predicted_mu):
        """
        Method for computing the KL divergence between the approximate posterior
        and the prior.

        Assumption is that the prior is N(0, I) and the approximate posterior
        is N(mu, sigma^2I).

        Parameters
        ----------
        predicted_sigma : tensorflow.Tensor
            The predicted (log) standard deviations of the latent distribution.
            Note: encoder returns log values of the standard deviations.
            Should have shape (None, latent_dim).
        predicted_mu : tensorflow.Tensor
            The predicted means of the latent distribution.
            Should have shape (None, latent_dim).

        Returns
        -------
        loss : float
            The loss for each data point. Will be a tensor of shape (None,).
        """
        # define a small minimum std. dev. to stabilize the computations
        k = tf.constant(1e-3, dtype=tf.float32)

        stds = tf.exp(predicted_sigma) + k
        stable_log_stds = tf.where(
            predicted_sigma < 0,
            tf.math.log(stds),
            predicted_sigma
        )

        if self.scaled_prior:
            prior_var = tf.constant(1.0/self.latent_dim, dtype=tf.float32)
        else:
            prior_var = tf.constant(1.0, dtype=tf.float32)

        mu_sum = tf.reduce_sum(tf.square(predicted_mu)/prior_var, axis=-1)
        var_sum = tf.reduce_sum(tf.square(stds)/prior_var, axis=-1)
        log_sum = tf.reduce_sum(
            2*stable_log_stds-tf.math.log(prior_var), axis=-1
        )
        
        loss = mu_sum + var_sum - log_sum

        return loss

    def distance_regularization(self, distances, true_properties):
        """
        Semantic distance regularization term. Penalizes large distances from
        properties that are not the true property, to keep the space from
        "blowing up".
        
        Essentially no loss for distances under a certain value (1.5) and 
        exponentially increasing penalty as distance grows after this.

        Parameters
        ----------
        distances : tensorflow.Tensor
            The distances to the prototypes.
        true_properties : tensorflow.Tensor
            The true properties (labels). One-hot encoded vectors.

        Returns
        -------
        d_reg : float
            The average distance regularization term over the batch of data.
        """
        others = tf.multiply(distances, (1.0-true_properties))
        penalties = (1/50)*others**5
        d_reg = tf.reduce_mean(penalties)
        return d_reg

    def reset_metrics(self):
        """
        Function to reset the metric-tracking objects.
        To be called at the end of each epoch.
        """
        self.training_accuracy.reset_states()
        self.validation_accuracy.reset_states()

    def get_config(self):
        config = super(DomainLearnerModel, self).get_config()
        config.update({
            'encoder_config': self.encoder.get_config() if self.encoder \
                else None,
            'decoder_config': self.decoder.get_config() if self.decoder \
                else None,
            'distance': self.dist,
            'similarity': self.sim,
            'similarity_c': self.sim_c,
            'alpha': self.alpha,
            'lambda': self.lam,
        })
        return config


# ------------------------------------------------------------------------------
    

class VariationalAutoencoder(tf.keras.models.Model):
    """
    This class defines a general end-to-end variational autoencoder model by
    taking in an already-defined encoder and decoder, as well as distribution
    information for both the latent space and the output.

    Assumptions: latent variables are iid, output variables are iid.
    """
    def __init__(
            self,
            encoder: tf.keras.models.Model,
            decoder: tf.keras.models.Model,
            latent_dimension: int,
            input_shape: Tuple[int],
            use_awgn: bool = False,
            awgn_variance: float = 1.0,
            **kwargs
        ):
        """
        Constructor method.

        Parameters
        ----------
        encoder : tf.keras.models.Model
            Encoder model.
            Note: needs to generate vectors that are 2*latent_dimension long.
            The first half of the vector represents the means, the second half
            represents the log standard deviations.
        decoder : tf.keras.models.Model
            Decoder model. Input is a vector of length latent_dimension.
        latent_dimension : int
            The dimension of the latent vector.
        input_shape : 3-element tuple
            Shape of the input, given as a tuple (height, width, channels).
        use_awgn : bool, optional
            Whether to use additive white Gaussian noise.
            Default value is False.
        awgn_variance : float, optional
            The variance of the Gaussian noise, if added.
            Default value is 1.0.
        """
        super(VariationalAutoencoder, self).__init__(**kwargs)

        self.latent_dim = latent_dimension
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        self.use_awgn = use_awgn
        self.awgn_variance = awgn_variance

        if self.use_awgn:
            self.awgn_layer = AWGNLayer(self.awgn_variance)
        self.encoder = encoder
        self.reparam_layer = ReparameterizationLayer(latent_dimension)
        self.decoder = decoder

    def compile(
            self,
            lam: float = 1.0,
            scaled_prior: bool = False,
            **kwargs
        ):
        """
        Function for compiling the model. This is where the loss functions and
        optimizer are defined.

        Parameters
        ----------
        lam : float, optional
            Hyperparameter controlling the scale of the KL divergence loss.
            This is the assumed variance for each of the output variables.
            Default value is 1.0.
        scaled_prior : bool, optional
            Whether to scale the loss term based on the prior by the latent
            dimension. Basically, this sets the prior distribution over the 
            latent features to be normal with 0 mean and 1/latent_dim variance.
        other arguments
            Other arguments to pass to tf.keras.models.Model.compile().
        """
        super(VariationalAutoencoder, self).compile(**kwargs)
        self.scaled_prior = scaled_prior
        self.lam = lam

    def call(self, inputs, training=False):
        if self.use_awgn:
            augmented = self.awgn_layer(inputs, training=training)
            latent_params = self.encoder(augmented)
        else:
            latent_params = self.encoder(inputs)
        log_stds, mus, zs = self.reparam_layer(latent_params)
        decoded = self.decoder(zs)
        return [log_stds, mus, zs, decoded]
    
    def train_step(self, data):
        x, y = data # Since this is an AE, x and y are the same
        with tf.GradientTape() as tape:
            pred_stds, pred_mus, _, pred_y = self(x, training=True)

            recon_term = self.reconstruction_loss(y, pred_y)
            kl_term = self.kl_divergence_loss(pred_stds, pred_mus)

            recon_loss = tf.reduce_mean(recon_term)
            kl_loss = tf.reduce_mean(kl_term)
            loss = recon_loss + self.lam*kl_loss

        # compute gradients and apply them
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": loss,
            "r_loss": recon_loss,
            "kl_term": kl_loss
        }

    def test_step(self, data):
        x, y = data # Since this is an AE, x and y are the same
        pred_stds, pred_mus, _, pred_y = self(x, training=False)

        recon_term = self.reconstruction_loss(y, pred_y)
        kl_term = self.kl_divergence_loss(pred_stds, pred_mus)
        
        recon_loss = tf.reduce_mean(recon_term)
        kl_loss = tf.reduce_mean(kl_term)
        loss = recon_loss + self.lam*kl_loss

        return {
            "loss": loss,
            "r_loss": recon_loss,
            "kl_term": kl_loss
        }

    def reconstruction_loss(self, y_true, y_pred):
        """
        Function for the reconstruction loss term of the ELBO.

        Basic assumption is that the output variables are conditionally 
        independent and Gaussian dsitributed given the latent vector. 
        Additionally, it is assumed that the variance is constant across the
        output variables. This variance is captured by the balancing parameter
        lam.

        Parameters
        ----------
        y_true : tensorflow.Tensor
            The true data (labels).
        y_pred : tensorflow.Tensor
            The reconstructed data. These are technically the means of the
            conditional Gaussian distributions.

        Returns
        -------
        loss : float
            The loss for each data point. Will be a tensor of shape (None,).
        """
        loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1,2,3])

        return loss

    def kl_divergence_loss(self, predicted_sigma, predicted_mu):
        """
        Method for computing the KL divergence between the approximate posterior
        and the prior.

        Parameters
        ----------
        predicted_sigma : tensorflow.Tensor
            The predicted (log) standard deviations of the latent distribution.
            Note: encoder returns log values of the standard deviations.
            Should have shape (None, latent_dim).
        predicted_mu : tensorflow.Tensor
            The predicted means of the latent distribution.
            Should have shape (None, latent_dim).

        Returns
        -------
        loss : float
            The loss for each data point. Will be a tensor of shape (None,).
        """
        # define a small minimum std. dev. to stabilize the computations
        k = tf.constant(1e-6, dtype=tf.float32)

        stds = tf.exp(predicted_sigma) + k
        stable_log_stds = tf.where(
            predicted_sigma < 0,
            tf.math.log(stds),
            predicted_sigma
        )

        if self.scaled_prior:
            prior_var = tf.constant(1.0/self.latent_dim, dtype=tf.float32)
        else:
            prior_var = tf.constant(1.0, dtype=tf.float32)

        mu_sum = tf.reduce_sum(tf.square(predicted_mu)/prior_var, axis=-1)
        var_sum = tf.reduce_sum(tf.square(stds)/prior_var, axis=-1)
        log_sum = tf.reduce_sum(
            2*stable_log_stds-tf.math.log(prior_var), axis=-1
        )
        
        loss = mu_sum + var_sum - log_sum

        return loss

    def get_config(self):
        config = super(VariationalAutoencoder, self).get_config()
        config.update({
            'encoder_config': self.encoder.get_config() if self.encoder else None,
            'decoder_config': self.decoder.get_config() if self.decoder else None,
            'rep_layer_config': self.reparam_layer.get_config(),
            'latent_dimension': self.latent_dim,
            'use_awgn': self.use_awgn,
            'awgn_variance': self.awgn_variance,
            'lam': self.lam,
            'height': self.height,
            'width': self.width,
            'channels': self.channels
        })
        return config
    

# ------------------------------------------------------------------------------