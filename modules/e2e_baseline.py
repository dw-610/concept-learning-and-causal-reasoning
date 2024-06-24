"""
This module contains code defining the end-to-end baseline model for testing
the SCCS-R system.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf
from keras import layers, models
from typing import Union

from modules.cslearn.controllers import ImageLearningController
from modules.causal_cs.decoder import SemanticDecoder

# ------------------------------------------------------------------------------

def get_encoder_model(latent_dim: int = 5, code_rate: float = 1.0/2.0,
                      stochastic_binarization: bool = True):
    """Get the encoder model for the E2E baseline system using CSLearn."""

    TRAIN_SET_SIZE: int     = 39209
    VALID_SET_SIZE: int     = 12630

    IM_SIZE: int            = 64
    NUM_CHANNELS: int       = 3
    NUM_CLASSES: int        = 4

    paths_dict: dict = {
        'train_data_path': 'local/sign-learning-data/signs_trn_data.npy',
        'train_labels_path': 'local/sign-learning-data/signs_trn_labels.npy',
        'valid_data_path': 'local/sign-learning-data/signs_vld_data.npy',
        'valid_labels_path': 'local/sign-learning-data/signs_vld_labels.npy'
    }

    shapes_dict: dict = {
        'train_data_shape': (TRAIN_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'train_labels_shape': (TRAIN_SET_SIZE, NUM_CLASSES),
        'valid_data_shape': (VALID_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'valid_labels_shape': (VALID_SET_SIZE, NUM_CLASSES)
    }

    BATCH_SIZE: int         = 64

    ARCH: str               = 'custom_cnn'
    GLOBAL_POOL_TYPE: str   = 'max'
    NUM_BLOCKS: int         = 5
    FILTERS: list           = [48, 96, 192, 384, 768]
    KERNEL_SIZES: list      = [3, 3, 3, 3, 3]
    STRIDES: list           = [2, 2, 2, 2, 2]
    DROPOUT: float          = 0.0

    ctrl = ImageLearningController(learner_type='classifier')
    ctrl.create_data_loaders(
        dataset='local',
        batch_size=BATCH_SIZE,
        paths_dict=paths_dict,
        shapes_dict=shapes_dict
    )
    ctrl.create_learner(
        latent_dim=latent_dim,
        architecture=ARCH,
        global_pool_type=GLOBAL_POOL_TYPE,
        number_of_blocks=NUM_BLOCKS,
        filters=FILTERS,
        kernel_sizes=KERNEL_SIZES,
        strides=STRIDES,
        dropout=DROPOUT
    )

    expanded_dim = int(latent_dim / code_rate)

    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = ctrl.encoder(inputs)
    x = layers.Dense(expanded_dim, activation='tanh', name='coder')(x)
    x = BinarizationLayer(stochastic_binarization)(x)

    encoder = models.Model(inputs=inputs, outputs=x)

    return encoder

# ------------------------------------------------------------------------------

def get_decoder_model(task: str, latent_dim: int = 5, n_classes: int = 4):
    """
    Function to get the decoder model for the baseline E2E system. Will just
    use the SemanticDecoder class from the causal_cs module to be consistent.
    """
    return SemanticDecoder(task, latent_dim, n_classes)

# ------------------------------------------------------------------------------

class BinarizationLayer(layers.Layer):
    """
    Custom layer that will binarize the input data. Data should be in the range
    [-1, 1] before passing through this layer, e.g. using tanh activation.
    """
    def __init__(self, stochastic: bool = True, **kwargs):
        """
        Constructor for the BinarizationLayer class.

        Parameters
        ----------
        stochastic : bool
            Whether to use stochastic binarization or not.
        """
        super(BinarizationLayer, self).__init__(**kwargs)
        self.stochastic = stochastic

    def call(self, inputs):
        if self.stochastic:
            random_numbers = tf.random.uniform(shape=tf.shape(inputs))
            probs   = (1 + inputs) / 2
            binary_values = tf.where(random_numbers < probs, 1., -1.)
        else:
            binary_values = tf.where(inputs >= 0, 1., -1.)
        return tf.stop_gradient(binary_values - inputs) + inputs
    
    def get_config(self):
        config = super(BinarizationLayer, self).get_config()
        config.update({'stochastic': self.stochastic})
        return config

# ------------------------------------------------------------------------------

class AWGNLayer(layers.Layer):
    """
    Custom layer that adds AWGN to the input data. The variance of the noise
    is ranomly chosen from the range [0.01, 100] during training, and is set
    to a specified value during testing. This corresponds to a SNR range of
    [20, -20] dB when E_b is set to 1.
    """
    def __init__(self, variance: float = 1.0, **kwargs):
        """
        Constructor for the AWGNLayer class. For use with BPSK modulation.

        Parameters
        ----------
        variance : float
            The variance of the AWGN to be added to the input data.
        """
        super(AWGNLayer, self).__init__(**kwargs)
        self.variance = variance

    def call(self, inputs, training=False): 
        if training:
            variance = tf.random.uniform(shape=()) * 99.99 + 0.01
        else:
            variance = self.variance
        noise = tf.random.normal(
            shape=tf.shape(inputs), 
            mean=0.0, 
            stddev=tf.sqrt(variance)
        )
        return noise + inputs

    # this is important for saving/loading the network
    def get_config(self):
        config = super(AWGNLayer, self).get_config()
        config.update({'variance': self.variance})
        return config
    
# ------------------------------------------------------------------------------

class ComplexAWGNLayer(layers.Layer):
    """
    Custom layer that adds complex AWGN to the input data, for use with QAM
    modulation schemes. The variance of the noise is ranomly chosen from the
    range [0.01, 100] during training, and is set to a specified value during
    testing. This corresponds to a SNR range of [20, -20] dB when E_s is 1.
    """
    def __init__(self, variance: float = 1.0, **kwargs):
        """
        Constructor for the ComplexAWGNLayer class. For use with QAM modulation.

        Parameters
        ----------
        variance : float
            The variance of the AWGN to be added to the input data.
        """
        super(ComplexAWGNLayer, self).__init__(**kwargs)
        self.variance = variance

    def call(self, inputs, training=False):
        if training:
            variance = tf.random.uniform(shape=()) * 99.99 + 0.01
        else:
            variance = self.variance
        noise = tf.random.normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=tf.sqrt(variance/2.0)
        )
        return noise + inputs

    # this is important for saving/loading the network
    def get_config(self):
        config = super(ComplexAWGNLayer, self).get_config()
        config.update({'variance': self.variance})
        return config

# ------------------------------------------------------------------------------

class QAM16ModulationLayer(layers.Layer):
    """
    Custom layer that implements 16QAM modulation on the input data. The input
    data should be in the form of bits, e.g. -1 or 1. The output data will be
    complex-valued, with the real and imaginary parts corresponding to the
    in-phase and quadrature components of the 16QAM symbols.
    """
    def __init__(self, **kwargs):
        super(QAM16ModulationLayer, self).__init__(**kwargs)

    @tf.custom_gradient
    def _compute_output(self, batch_ints):
        """Custom gradient function to allow gradients to flow."""
        def compute_values(x):
            return tf.where(x == 0, -3.0,
                    tf.where(x == 1, -1.0,
                    tf.where(x == 2,  3.0,
                             1.0)))

        batch_real = compute_values(batch_ints[:,0::2])
        batch_imag = compute_values(batch_ints[:,1::2])

        output = tf.concat([batch_real, batch_imag], axis=1) / (10**0.5)

        def custom_grad(dy):
            return dy

        return output, custom_grad

    def call(self, inputs):
        batch_bits = (inputs + 1) / 2
        batch_ints = 2 * batch_bits[:, 0::2] + batch_bits[:, 1::2]
        return self._compute_output(batch_ints)

    def get_config(self):
        return super(QAM16ModulationLayer, self).get_config()

# ------------------------------------------------------------------------------

class EndToEndSemanticModel(models.Model):
    """
    The end-to-end semantic baseline for comparing with the SCCS-R system.
    """
    def __init__(
            self,
            tasks: Union[str, list],
            latent_dim: int,
            n_classes: Union[int, list],
            n_decoders: int = 1,
            stoch_binarization: bool = True,
            mod_type: str = 'BPSK',
            code_rate: float = 0.5
        ):
        """
        Constructor for the EndToEndSemanticModel class.

        Parameters
        ----------
        tasks : str or list
            The task(s) to perform with the decoders. If more than one decoder,
            this MUST be a list of the tasks for each decoder.
        latent_dim : int
            The dimension of the latent space for the encoder.
        n_classes : int or list
            The number of classes for the decoder(s). If more than one decoder,
            this MUST be a list of the number of classes for each decoder.
        n_decoders : int
            The number of decoders to use in the system. 
            Default is 1.
        stoch_binarization : bool
            Whether to use stochastic binarization or not.
            Default is True.
        mod_type : str
            The modulation type to use. Options are 'BPSK' and '16-QAM'.
            Default is 'BPSK'.
        code_rate : float
            The "code rate" of the system. Will expand the latent dimension by
            the whole-number inverse of this factor to get the encoder output.
            Default is 0.5.
        tasks : list
            The tasks to perform with the decoders. 
        """
        super(EndToEndSemanticModel, self).__init__()

        if n_decoders > 1:
            if isinstance(n_classes, int):
                raise ValueError(
                    'If using multiple decoders, n_classes must be a list.')
            if isinstance(tasks, str):
                raise ValueError(
                    'If using multiple decoders, task must be a list.')
            if len(n_classes) != n_decoders:
                raise ValueError('Length of n_classes must match n_decoders.')
            if len(tasks) != n_decoders:
                raise ValueError('Length of tasks must match n_decoders.')
        elif n_decoders < 1:
            raise ValueError('n_decoders must be at least 1.')
        
        self._n_decoders = n_decoders

        expanded_dim = int(latent_dim / code_rate)

        self.encoder = get_encoder_model(
            latent_dim, code_rate, stoch_binarization)

        if mod_type == 'BPSK':
            self.awgn = AWGNLayer()
            self.decoders = []
            for task, n in zip(tasks, n_classes):
                self.decoders.append(get_decoder_model(task, expanded_dim, n))
            self.modulator = None
        elif mod_type == '16-QAM':
            self.modulator = QAM16ModulationLayer()
            self.awgn = ComplexAWGNLayer()
            if n_decoders == 1:
                self.decoder = get_decoder_model(
                    tasks, int(expanded_dim/2), n_classes)
            else:
                self.decoders = []
                for tasks, n in zip(tasks, n_classes):
                    self.decoders.append(get_decoder_model(
                        tasks, int(expanded_dim/2), n))
        else:
            raise ValueError('Invalid modulation type.')
        
    def call(self, inputs, training=False):
        x = self.encoder(inputs)

        if self.modulator is not None:
            x = self.modulator(x)
            x = self.awgn(x, training=training)
        else:
            x = self.awgn(x, training=training)

        if self._n_decoders == 1:
            return self.decoder(x)
        else:
            return [decoder(x) for decoder in self.decoders]
    
    def summary(self):
        x = tf.keras.Input(shape=(64, 64, 3))
        model = tf.keras.models.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
    def get_config(self):
        config = super(EndToEndSemanticModel, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'stoch_binarization': self.stoch_binarization,
            'mod_type': self.mod_type,
            'code_rate': self.code_rate
        })
        return config
        
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    from keras import backend

    QAM16 = False

    if QAM16:
        mod = 'QAM16'
    else:
        mod = 'BPSK'

    latent_dim = 4
    n_classes = 4

    test_data = tf.random.normal((3, 64, 64, 3))

    encoder = get_encoder_model(latent_dim)
    x = encoder(test_data)
    print('\nEncoder op:')
    print(x)
    
    if QAM16:
        x = QAM16ModulationLayer()(x)
        print('\nQAM16 op:')
        print(x)
        x = ComplexAWGNLayer(0.1)(x)
        print('\nAWGN op:')
        print(x)
    else:
        x = AWGNLayer(0.1)(x)
        print('\nAWGN op:')
        print(x)

    decoder_dim = latent_dim if not QAM16 else int(latent_dim/2)

    decoder = get_decoder_model(decoder_dim, n_classes)
    print('\nDecoder op:')
    x = decoder(x)
    print(x)

    backend.clear_session()

    model = EndToEndSemanticModel(latent_dim, n_classes, True, mod, 1./32)
    model(test_data)
    print('\nModel summary:')
    model.summary()
    