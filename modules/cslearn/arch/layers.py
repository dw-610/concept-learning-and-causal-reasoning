"""
This module contains the layers for creating custom model architectures. Each
class is a subclass of tf.keras.layers.Layer.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf
import math

layers = tf.keras.layers
initializers = tf.keras.initializers

# ------------------------------------------------------------------------------


class AWGNLayer(layers.Layer):
    """
    Custom layer to add additive white Gaussian noise to the given input.

    Only adds noise during training.
    """
    def __init__(
            self, 
            variance: float = 1.0
        ):
        """
        Constructor method for AWGNLayer.

        Parameters
        ----------
        variance : float, optional
            Variance of the IID AWGN that is added to the input.
            Default value is 1.0.
        """
        super(AWGNLayer, self).__init__()
        self.var  = variance

    def call(self, input, training=False):
        if not training:
            return input
        noise = tf.random.normal(
            shape=tf.shape(input), 
            mean=0., 
            stddev=tf.sqrt(self.var)
        )
        return input + noise

    # this is important for saving/loading the network
    def get_config(self):
        return super(AWGNLayer, self).get_config()
    
    def compute_output_shape(self, input_shape):
        return input_shape
    

# ------------------------------------------------------------------------------


class EuclideanDistanceLayer(layers.Layer):
     """
     This custom layer uses a list of prototype vectors to compute the 
     Euclidean distance. Input to the layer is a single vector in the 
     conceptual space and the output is a vector of distances to the prototypes.

     The prototypes argument is tf.Variable, and is typically updated
     iteratively throughout the training process.

     See D. Wheeler, B. Natarajan, "Autoencoder-Based Domain Learning for 
     Semantic Communication with Conceptual Spaces," 2024 for more info on this.
     """
     def __init__(
               self,
               prototypes: tf.Variable
          ):
          """
          Constructor method for EuclideanDistanceLayer.

          Parameters
          ----------
          prototypes : tf.Variable
               The initial prototype tensor. Should be 2D, where rows represent
               prototypes and column represent the dimensions.
          """
          super(EuclideanDistanceLayer, self).__init__()
          self.prototypes = prototypes
          self.input_dim = self.prototypes.shape[1]

     def build(self, input_shape):
        super(EuclideanDistanceLayer, self).build(input_shape)

     def call(self, x, training=False):
          # Compute differences between each input and each prototype
          diff = tf.expand_dims(x, axis=1) - self.prototypes
          
          # Compute distances
          out = tf.sqrt(tf.einsum('bij,bij->bi', diff, diff))
          return out

     def compute_output_shape(self, input_shape):
          return (input_shape[0], self.prototypes.shape[0])


# ------------------------------------------------------------------------------


class GaussianSimilarityLayer(layers.Layer):
    """
    This custom layer computes Gaussian similarity given some distances.
    Input to the layer is a single vector of distances,
    and the output is a vector of similarities to the prototypes.

    The Gaussian similarity operates on distance values and is defined as
    exp(-c*d^2), where c is a constant and d is the distance. Similarity scores
    are from 0-1, and higher c values result in a sharper Gaussian curve.
    """
    def __init__(
            self,
            c: float = 1.0
        ):
        """
        Constructor method for GaussianSimilarityLayer.

        Parameters
        ----------
        c : float, optional
            The constant in the Gaussian similarity function.
            Default value is 1.0.
        """
        super(GaussianSimilarityLayer, self).__init__()
        self.c = c
        
    def build(self, input_shape):
        super(GaussianSimilarityLayer, self).build(input_shape)

    def call(self, x):
        out = tf.exp(-self.c*x**2)
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape
    

# ------------------------------------------------------------------------------


class SoftGaussSimPredictionLayer(layers.Layer):
    """
    This custom layer makes maximimum-similarity decisions based on distances
    to prototypes. Input to the layer is a single (batch of) vector(s) of
    distances between features and prototypes. The output is a single
    (batch of) vector(s) of normalized similarities to the prototypes, for
    classification. The normalized similarities are softmax-esque.

    Essentially implements a "soft-min" of the distances by way of a Gaussian
    similarity function, with a numerical stability trick and a tuning parameter
    c. Equivalent to taking the softmax of -c*d^2.
    """
    def __init__(
            self,
            c: float = 1.0
        ):
        """
        Constructor method for SoftGaussSimPredictionLayer.

        Parameters
        ----------
        c : float, optional
            The constant in the Gaussian similarity function.
            Default value is 1.0.
        """
        super(SoftGaussSimPredictionLayer, self).__init__()
        self.c = c
        
    def build(self, input_shape):
        super(SoftGaussSimPredictionLayer, self).build(input_shape)

    def call(self, x):
        # compute the maximum of -c*d^2, subtract for stability
        max_vals = tf.reduce_max(-self.c*x**2, axis=1, keepdims=True)
        norm_x = -self.c*x**2 - max_vals

        # compute the Gaussian-similarity proportional probabilities
        out = tf.exp(norm_x)/tf.reduce_sum(tf.exp(norm_x),axis=1,keepdims=True)
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape


# ------------------------------------------------------------------------------


class ConvolutionBlock(layers.Layer):
    """
    This custom layer implements a convolutional block with a conv2D layer with
    batch normalization and optional maxpooling.
    """
    def __init__(
            self,
            filters: int,
            kernel_size: int = 3, 
            strides: int = 1, 
            activation: str = 'relu',
            maxpool: bool = False
        ):
        """
        Constructor method.

        Parameters
        ----------
        filters : int     
            The number of filters to use in this block.
        kernel_size : int, optional   
            Length/width of the convolutional kernel.
            Default value is 3.
        strides : int, optional      
            How many "steps" kernel takes in each step of the convolution.
            Default value is 1.
        activation : str, optional  
            Activation function applied after convolution.
            Default value is 'relu'.
        maxpool : bool, optional      
            Whether 2x2 maxpooling is applied after convolution.
            Default value is False.
        """
        super(ConvolutionBlock, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.use_maxpool = maxpool

        self.conv = layers.Conv2D(
            filters, 
            kernel_size=kernel_size, 
            strides=strides,
            padding='same'
        )

        self.batch_norm = layers.BatchNormalization()

        self.act = layers.Activation(activation=activation)

        self.maxpool = layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        ) if maxpool else None

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        x = self.batch_norm(x, training=training)
        x = self.act(x, training=training)
        if self.use_maxpool:
            x = self.maxpool(x, training=training)
        return x
    
    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self.activation,
            'maxpool': self.use_maxpool
        })
        return config
    
    def compute_output_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        if self.strides > 1:
            height = math.ceil(height/self.strides)
            width = math.ceil(width/self.strides)
        if self.use_maxpool:
            height = math.ceil(height/2)
            width = math.ceil(width/2)
        return (input_shape[0], height, width, self.filters)
        
    
# ------------------------------------------------------------------------------


class DeconvolutionBlock(layers.Layer):
    """
    This custom layer implements a deconvolutional block with a conv2DTranspose
    with batch normalization and optional 'unpooling' (inverse of maxpooling).
    """
    def __init__(
            self,
            filters: int, 
            kernel_size: int = 3, 
            strides: int = 1, 
            activation: str = 'relu',
            unpool: bool = False
        ):
        """
        Constructor method for DeconvolutionBlock.

        Parameters
        ----------
        filters : int       
            Number of filters to use in this block.
        kernel_size : int, optional  
            Length/width of the deconvolutional kernel.
            Default value is 3.
        strides : int, optional     
            How many "steps" the kernel takes in each step of the deconvolution.
            Default value is 1.
        activation : str, optional    
            Activation function applied after deconvolution.
            Default value is 'relu'.
        unpool : bool, optional       
            Whether 2x2 maxpooling needs to be "undone".
            Default value is False.
        """

        super(DeconvolutionBlock, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.use_unpool = unpool

        self.unpool = layers.Conv2DTranspose(
            filters, 
            kernel_size=1, 
            strides=(2,2),
            padding='same'
        ) if unpool else None

        self.deconv = layers.Conv2DTranspose(
            filters, 
            kernel_size=kernel_size, 
            strides=strides,
            padding='same'
        )

        self.batch_norm = layers.BatchNormalization()

        self.act = layers.Activation(activation=activation)

    def call(self, x, training=False):  
        if self.use_unpool:
            x = self.unpool(x, training=training)
        x = self.deconv(x, training=training)
        x = self.batch_norm(x, training=training)
        x = self.act(x, training=training)

        return x
    
    def get_config(self):
        config = super(DeconvolutionBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self.activation,
            'unpool': self.use_unpool
        })
        return config
    
    def compute_output_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        height = height*self.strides
        width = width*self.strides
        if self.use_unpool:
            height = height*2
            width = width*2
        return (input_shape[0], height, width, self.filters)
    

# ------------------------------------------------------------------------------


class HeightWidthSliceLayer(layers.Layer):
    """
    A very simple layer which slices the input tensor along the height and width
    dimensions. Needed in the ConvDecoder model to line up dimensions in case
    where an odd-sized image is downsampled in the encoder.
    """
    def __init__(
            self,
            to_slice: int = 1
        ):
        """
        Constructor method for the HeightWidthSliceLayer.

        Parameters
        ----------
        to_slice : int, optional
            The number of dimensions to slice off the input tensor.
            Default value is 1.
        """
        super(HeightWidthSliceLayer, self).__init__()
        self.to_slice = to_slice

    def call(self, x, training=False):
        return x[:, :-self.to_slice, :-self.to_slice, :]
    
    def get_config(self):
        config = super(HeightWidthSliceLayer, self).get_config()
        config.update({
            'to_slice': self.to_slice
        })
        return config
    
    def compute_output_shape(self, input_shape):
        return (
            input_shape[0], 
            input_shape[1]-self.to_slice, 
            input_shape[2]-self.to_slice, 
            input_shape[3]
        )
    

# ------------------------------------------------------------------------------


class SmallResNetBlock(layers.Layer):
    """
    This custom layer implements the smaller residual block used in the ResNet18
    and ResNet34 models.

    See Table 1 of "Deep Residual Learning for Image Recognition" by He et al.
    for details.
    """
    def __init__(
            self,
            filters: int,
            activation: str = 'relu',
            downsample: bool = False
        ):
        """
        Constructor method.

        Parameters
        ----------
        x : tensorflow.Tensor             
            Input tensor to the block.
        filters : int     
            The number of filters to use in this block.
        activation : str, optional  
            Activation function applied after the block.
            Default value is 'relu'.
        downsample : bool, optional
            Whether to downsample the input.
            Default value is False.
        """
        super(SmallResNetBlock, self).__init__()

        self.filters = filters
        self.activation = activation
        self.downsample = downsample

        # define the layers in the residual path
        self.conv1 = layers.Conv2D(
            filters=int(filters), 
            kernel_size=3, 
            strides=2 if downsample else 1,
            padding='same'
        )
        self.batch_norm1 = layers.BatchNormalization()
        self.act1 = layers.Activation(activation=activation)

        self.conv2 = layers.Conv2D(
            filters = int(filters),
            kernel_size = 3,
            strides = 1,
            padding = 'same'
        )
        self.batch_norm2 = layers.BatchNormalization()

        # if downsampling, define layer to resize the input
        if downsample:
            self.resize = layers.Conv2D(
                filters = int(filters),
                kernel_size = 1,
                strides = 2,
                padding = 'same'
            )

        # define the adding layer
        self.add = layers.Add()

        # define the activation layer
        self.act_both = layers.Activation(activation=activation)

    def call(self, x, training=False):

        # residual path
        r = self.conv1(x, training=training)
        r = self.batch_norm1(r, training=training)
        r = self.act1(r, training=training)

        r = self.conv2(r, training=training)
        r = self.batch_norm2(r, training=training)

        # if downsampling, include the resize layer
        if self.downsample:
            x = self.resize(x, training=training)

        # add the residual to the input
        x = self.add([r,x])

        # activation
        x = self.act_both(x, training=training)

        return x
    
    def get_config(self):
        config = super(SmallResNetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'activation': self.activation,
            'downsample': self.downsample
        })
        return config
    
    def compute_output_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        if self.downsample:
            height = math.ceil(height/2)
            width = math.ceil(width/2)
        return (input_shape[0], height, width, self.filters)
    

# ------------------------------------------------------------------------------


class SmallDeResNetBlock(layers.Layer):
    """
    This custom layer implements a de-convolutional version of the smaller 
    residual block used in the ResNet18 and ResNet34 models. Built to mirror
    the operations of the SmallResNetBlock layer.
    """
    def __init__(
            self,
            filters: int,
            activation: str = 'relu',
            upsample: bool = False
        ):
        """
        Constructor method.

        Parameters
        ----------
        filters : int     
            The number of filters to use in this block.
        activation : str, optional  
            Activation function applied after the block.
            Default value is 'relu'.
        upsample : bool, optional
            Whether to upsample the input.
            Default value is False.
        """

        super(SmallDeResNetBlock, self).__init__()

        self.filters = filters
        self.activation = activation
        self.upsample = upsample

        # define the layers in the residual path
        self.deconv1 = layers.Conv2DTranspose(
            filters=int(filters), 
            kernel_size=3, 
            strides=1,
            padding='same'
        )
        self.batch_norm1 = layers.BatchNormalization()
        self.act1 = layers.Activation(activation=activation)

        self.deconv2 = layers.Conv2DTranspose(
            filters = int(filters/2) if upsample else int(filters),
            kernel_size = 3,
            strides = 2 if upsample else 1,
            padding = 'same'
        )
        self.batch_norm2 = layers.BatchNormalization()

        # if downsampling, define layer to resize the input
        if upsample:
            self.resize = layers.Conv2DTranspose(
                filters = int(filters/2) if upsample else int(filters),
                kernel_size = 1,
                strides = 2,
                padding = 'same'
            )

        # define the adding layer
        self.add = layers.Add()

        # define the activation layer
        self.act_both = layers.Activation(activation=activation)

    def call(self, x, training=False):

        # residual path
        r = self.deconv1(x, training=training)
        r = self.batch_norm1(r, training=training)
        r = self.act1(r, training=training)

        r = self.deconv2(r, training=training)
        r = self.batch_norm2(r, training=training)

        # if upsampling, include the resize layer
        if self.upsample:
            x = self.resize(x, training=training)

        # add the residual to the input
        x = self.add([r,x])

        # activation
        x = self.act_both(x, training=training)

        return x
    
    def get_config(self):
        config = super(SmallDeResNetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'activation': self.activation,
            'upsample': self.upsample
        })
        return config
    
    def compute_output_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        if self.upsample:
            height = height*2
            width = width*2
        return (input_shape[0], height, width, self.filters)
    

# ------------------------------------------------------------------------------


class ResNetBlock(layers.Layer):
    """
    This custom layer implements the residual block used in ResNet50, ResNet101,
    and ResNet152.

    See Table 1 of "Deep Residual Learning for Image Recognition" by He et al.
    for details.
    """
    def __init__(
            self,
            filters: int,
            activation: str = 'relu', 
            downsample: bool = False,
            match_filters: bool = False
        ):
        """
        Constructor method.

        Parameters
        ----------
        filters : int     
            The number of filters to use in this block.
        activation : str, optional  
            Activation function applied after the block.
            Default value is 'relu'.
        downsample : bool, optional
            Whether to downsample the input.
            Default value is False.
        match_filters : bool, optional
            Whether to match the number of filters in the residual path.
            Default value is False.
        """

        super(ResNetBlock, self).__init__()

        self.filters = filters
        self.activation = activation
        self.downsample = downsample
        self.match_filters = match_filters

        # define the layers in the residual path
        self.conv1 = layers.Conv2D(
            filters=int(filters/4), 
            kernel_size=1, 
            strides=2 if downsample else 1,
            padding='same'
        )
        self.batch_norm1 = layers.BatchNormalization()
        self.act1 = layers.Activation(activation=activation)

        self.conv2 = layers.Conv2D(
            filters=int(filters/4), 
            kernel_size=3, 
            strides=1,
            padding='same'
        )
        self.batch_norm2 = layers.BatchNormalization()
        self.act2 = layers.Activation(activation=activation)

        self.conv3 = layers.Conv2D(
            filters = int(filters),
            kernel_size = 1,
            strides = 1,
            padding = 'same'
        )
        self.batch_norm3 = layers.BatchNormalization()

        # if downsampling, define layer to resize the input
        if downsample:
            self.resize = layers.Conv2D(
                filters = int(filters),
                kernel_size = 1,
                strides = 2,
                padding = 'same'
            )
        elif match_filters:
            self.resize = layers.Conv2D(
                filters = int(filters),
                kernel_size = 1,
                strides = 1,
                padding = 'same'
            )

        # define the adding layer
        self.add = layers.Add()

        # define the activation layer
        self.act_both = layers.Activation(activation=activation)

    def call(self, x, training=False):

        # residual path
        r = self.conv1(x, training=training)
        r = self.batch_norm1(r, training=training)
        r = self.act1(r, training=training)

        r = self.conv2(r, training=training)
        r = self.batch_norm2(r, training=training)
        r = self.act2(r, training=training)

        r = self.conv3(r, training=training)
        r = self.batch_norm3(r, training=training)

        # if downsampling include resize layer, else make sure the filters match
        if self.downsample:
            x = self.resize(x, training=training)
        elif self.match_filters:
            x = self.resize(x, training=training)

        # add the residual to the input
        x = self.add([r,x])

        # activation
        x = self.act_both(x, training=training)

        return x
    
    def get_config(self):
        config = super(ResNetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'activation': self.activation,
            'downsample': self.downsample,
            'match_filters': self.match_filters
        })
        return config
    
    def compute_output_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        if self.downsample:
            height = math.ceil(height/2)
            width = math.ceil(width/2)
        return (input_shape[0], height, width, self.filters)
    

# ------------------------------------------------------------------------------


class DeResNetBlock(layers.Layer):
    """
    This custom layer implements a de-convolutional version of the typical
    residual block used in the ResNett50 and larger models. Built to mirror
    the operations of the ResNetBlock layer.
    """
    def __init__(
            self,
            filters: int,
            activation: str = 'relu', 
            upsample: bool = False,
        ):
        """
        Constructor method.

        Parameters
        ----------
        filters : int     
            The number of filters to use in this block.
        activation : str, optional  
            Activation function applied after the block.
            Default value is 'relu'.
        upsample : bool, optional
            Whether to upsample the input.
            Default value is False.
        """

        super(DeResNetBlock, self).__init__()

        self.filters = filters
        self.activation = activation
        self.upsample = upsample

        # define the layers in the residual path
        self.deconv1 = layers.Conv2DTranspose(
            filters=int(filters/4), 
            kernel_size=1, 
            strides=1,
            padding='same'
        )
        self.batch_norm1 = layers.BatchNormalization()
        self.act1 = layers.Activation(activation=activation)

        self.deconv2 = layers.Conv2DTranspose(
            filters=int(filters/4), 
            kernel_size=3, 
            strides=1,
            padding='same'
        )
        self.batch_norm2 = layers.BatchNormalization()
        self.act2 = layers.Activation(activation=activation)

        if self.upsample:
            self.deconv3 = layers.Conv2DTranspose(
                filters = int(filters/2),
                kernel_size = 1,
                strides = 2,
                padding = 'same'
            )
        else:
            self.deconv3 = layers.Conv2DTranspose(
                filters = int(filters),
                kernel_size = 1,
                strides = 1,
                padding = 'same'
            )
        self.batch_norm3 = layers.BatchNormalization()

        # if upsampling, define layer to resize the input
        if upsample:
            self.resize = layers.Conv2DTranspose(
                filters = int(filters/2) if upsample else int(filters),
                kernel_size = 1,
                strides = 2,
                padding = 'same'
            )

        # define the adding layer
        self.add = layers.Add()

        # define the activation layer
        self.act_both = layers.Activation(activation=activation)

    def call(self, x, training=False):

        # residual path
        r = self.deconv1(x, training=training)
        r = self.batch_norm1(r, training=training)
        r = self.act1(r, training=training)

        r = self.deconv2(r, training=training)
        r = self.batch_norm2(r, training=training)
        r = self.act2(r, training=training)

        r = self.deconv3(r, training=training)
        r = self.batch_norm3(r, training=training)

        # if upsampling, expand the input
        if self.upsample:
            x = self.resize(x, training=training)

        # add the residual to the input
        x = self.add([r,x])

        # activation
        x = self.act_both(x, training=training)

        return x
    
    def get_config(self):
        config = super(DeResNetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'activation': self.activation,
            'upsample': self.upsample
        })
        return config
    
    def compute_output_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        if self.upsample:
            height = height*2
            width = width*2
        return (input_shape[0], height, width, self.filters)
    

# ------------------------------------------------------------------------------
    

class ReparameterizationLayer(layers.Layer):
    """
    Layer used in the variational autoencoder to allow for the flow of gradients
    through the reparameterization trick. Takes a single vector where the first
    half of the dimensions are the means and the second half are the log
    standard deviations. Uses these to sample a latent vector from a normal
    distribution and returns the log standard deviations, means, and the latent
    representation.

    Right now, it is just hard-coded that eps ~ N(0,I) and z ~ N(mu,diag(sig^2))
    where mu and log(sig) are given by the input vector.
    """
    def __init__(self, latent_dim: int):
        super(ReparameterizationLayer, self).__init__()
        self.latent_dim = latent_dim

    def call(self, x, training=False):
        # Pull out the means and (log) standard devs
        mus = x[:,:self.latent_dim]
        log_stds = x[:,self.latent_dim:]

        # sample random vectors for the generating the latent vector
        eps = tf.random.normal(shape=tf.shape(mus))

        # compute the latent vectors
        zs = mus + tf.multiply(eps,tf.exp(log_stds))

        return [log_stds, mus, zs]

    def get_config(self):
        config = super(ReparameterizationLayer, self).get_config()
        config.update({
            'latent_dim': self.latent_dim
        })
        return config

    def compute_output_shape(self, input_shape):
        dim = input_shape[1]
        new_dim = int(dim/2)
        return (input_shape[0], new_dim)
    

# ------------------------------------------------------------------------------