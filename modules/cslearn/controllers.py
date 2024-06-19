"""
This module wraps the functionality of the other modules, where the classes 
defined are used as the primary interface for training and evaluation.
"""

# ------------------------------------------------------------------------------
# imports

try:
    from comet_ml import Experiment
    comet_imported = True
except ImportError:
    print("Warning: comet_ml not installed. Comet ML logging will not be" \
        + " available.")
    comet_imported = False

import gc
import os
import json
import datetime

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import Optional, Union
from contextlib import nullcontext

from . import feeders
from . import utilities as utils
from . import visualization as vis
from . import training as cstrain
from .arch import models, layers

# ------------------------------------------------------------------------------

class ImageLearningController():
    """
    This class integrates functionality accross the other modules and is used as
    the primary interface for training models operating on image data.
    """

    # --- attributes ---
    # v v v v v v v v v v v v v 

    learner_type = None         # Type of model to train (e.g., 'classifier')
    dataset = None              # Dataset identifier (e.g., 'mnist')
    architecture = None         # Model architecture (e.g., 'custom_cnn')
    latent_dim = None           # Dimension of the latent space
    batch_size = 32             # Batch size for training
    buffer_size = 10000         # Buffer size for training
    autoencoder_type = 'standard' # Type of autoencoder to use

    model = None                # The main model object
    encoder = None              # The encoder model object
    decoder = None              # The decoder model object

    wass_train = False          # Whether to use the Wasserstein loss

    debug_mode = False          # Whether to use the debug mode
    use_gpu = True              # Whether to use the GPU for model training

    height = None               # Height of the input images
    width = None                # Width of the input images
    channels = None             # Number of channels in the input images
    number_of_properties = None # Number of semantic properties in the dataset

    params = {}                 # Dict of parameters for loading and logging

    def __init__(
            self,
            learner_type: str,
            use_gpu: Optional[bool] = True,
            debug: Optional[bool] = False,
        ):
        """
        Constructor for the ImageLearningController class.

        Parameters
        ----------
        learner_type : str
            The type of learner to train. Options are 'classifier',
                'autoencoder', 'domain_learner'.
        debug : bool, optional
            Whether to use the debug mode.
            Default is False.
        """

        assert learner_type in ['classifier', 'autoencoder', \
                                'domain_learner'], \
            "Model type must be one of 'classifier', 'autoencoder', " \
                + "'domain_learner'."

        self.learner_type = learner_type

        self.params['learner_type'] = learner_type

        self.data_loaders_created = False
        self.encoder_created = False
        self.decoder_created = False
        self.models_created = False
        self.models_compiled = False
        self.models_trained = False
        self.encoder_loaded = False
        self.decoder_loaded = False
        self.models_loaded = False

        if debug:
            self.debug_mode = True
            tf.config.run_functions_eagerly(True)

        if not use_gpu:
            self.use_gpu = False
            tf.config.set_visible_devices([], 'GPU')

    # --------------------- user-facing methods --------------------------------
    # v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v 

    def create_data_loaders(
            self,
            dataset: str,
            batch_size: Optional[int] = 32,
            buffer_size: Optional[int] = 10000,
            paths_dict: Optional[dict] = None,
            shapes_dict: Optional[dict] = None
        ):
        """
        This method creates the data loaders for the model. The data loaders
        are created based on the dataset specified and are saved as attributes 
        of the class. Data loaders are tf.Data.Dataset objects.

        Parameters
        ----------
        dataset : str
            The name of the dataset to load in. Options are 'mnist', 'cifar10',
            or 'local'. If 'local' is specified, then data is loaded in from
            the provided paths as np.memmap arrays.
        batch_size : int, optional
            The batch size to use for training.
            Default value is 32.
        buffer_size : int, optional
            The random buffer size to use for training.
            Default value is 10000.
        paths_dict : dict, optional
            A dictionary containing the paths to the data and labels arrays, if
            using 'local' data. For 'classifier' and 'domain_learner' models,
            the dictionary should be of the form:
            {
                'train_data_path': 'path/to/train_data.npy',
                'train_labels_path': 'path/to/train_labels.npy',
                'valid_data_path': 'path/to/train_data.npy',
                'valid_labels_path': 'path/to/train_labels.npy'
            }
            For 'autoencoder' models, the dictionary should be of the form:
            {
                'train_data_path': 'path/to/train_data.npy',
                'valid_data_path': 'path/to/train_data.npy'
            }
            Default is None.
        shapes_dict : dict, optional
            A dictionary containing the shapes of the data and labels arrays, if
            using 'local' data.
            For 'classifier' and 'domain_learner' models, the dictionary should
            be of the form:
            {
                'train_data_shape': (train_samples, height, width, channels),
                'train_labels_shape': (train_samples, number_of_properties),
                'valid_data_shape': (valid_samples, height, width, channels),
                'valid_labels_shape': (valid_samples, number_of_properties)
            }
            For 'autoencoder' models, the dictionary should be of the form:
            {
                'train_data_shape': (train_samples, height, width, channels),
                'valid_data_shape': (valid_samples, height, width, channels)
            }
            Default is None.
        """

        # do some parameter checking
        if dataset == 'local':
            if paths_dict is None or shapes_dict is None:
                raise ValueError("Must provide paths_dict and shapes_dict for" \
                    + " 'local' dataset.")
                
        # save off important values as attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        if dataset == 'mnist':
            self.height = 28
            self.width = 28
            self.channels = 1
            self.number_of_properties = 10
            self.train_size = 60000
            self.valid_size = 10000
        elif dataset == 'cifar10':
            self.height = 32
            self.width = 32
            self.channels = 3
            self.number_of_properties = 10
            self.train_size = 50000
            self.valid_size = 10000
        elif dataset == 'local':
            self.height = shapes_dict['train_data_shape'][1]
            self.width = shapes_dict['train_data_shape'][2]
            self.channels = shapes_dict['train_data_shape'][3]
            self.number_of_properties = shapes_dict['train_labels_shape'][1]
            self.train_size = shapes_dict['train_data_shape'][0]
            self.valid_size = shapes_dict['valid_data_shape'][0]
        else:
            raise ValueError("Got invalid dataset.")

        # save these value in the params dictionary
        self.params['dataset'] = dataset
        self.params['batch_size'] = batch_size
        self.params['buffer_size'] = buffer_size
        self.params['height'] = self.height
        self.params['width'] = self.width
        self.params['channels'] = self.channels
        self.params['number_of_properties'] = self.number_of_properties
                
        # get the appropriate internal method for creating the data loaders
        if self.learner_type == 'classifier':
            loader_method = self._create_classifier_data_loaders
        elif self.learner_type == 'autoencoder':
            loader_method = self._create_autoencoder_data_loaders
        elif self.learner_type in ['domain_learner','space_learner']:
            loader_method = self._create_semantic_data_loaders
        else:
            raise ValueError("Got invalid learner type.")

        # call the internal method for creating the data loaders
        loader_method(
            dataset=dataset,
            batch_size=batch_size,
            buffer_size=buffer_size,
            paths_dict=paths_dict,
            shapes_dict=shapes_dict
        )

        # set flag to indicate that data loaders have been created
        self.data_loaders_created = True
            
    def create_learner(
            self,
            latent_dim: int,
            architecture: Optional[str] = 'custom_cnn',
            autoencoder_type: Optional[str] = 'standard',
            number_of_blocks: Optional[int] = 4,
            filters: Optional[Union[list,int]] = [16,16,32,32],
            kernel_sizes: Optional[Union[list,int]] = [7,5,3,3],
            strides: Optional[Union[list,int]] = [2,1,2,1],
            use_maxpool: Optional[Union[list,bool]] = False,
            hidden_activation: Optional[str] = 'relu',
            latent_activation: Optional[str] = 'linear',
            output_activation: Optional[str] = 'linear',
            global_pool_type: Optional[str] = 'avg',
            dropout: Optional[float] = None,
            use_awgn: Optional[bool] = False,
            awgn_variance: Optional[float] = 0.1,
            distance: Optional[str] = 'euclidean',
            similarity: Optional[str] = 'gaussian',
            similarity_c: Optional[float] = 1.0,
            initial_protos: Optional[np.ndarray] = None
        ):
        """
        This method creates the models for the learning task. For the image 
        learning tasks, all models are based on convolutional neural networks.
        Models are realized as tf.keras.Model objects.

        Parameters
        ----------
        latent_dim : int
            The dimension of the latent space.
        architecture : str, optional
            The architecture of the model. Options are 'custom_cnn', 'resnet18',
            'resnet34', or 'resnet50'.
            Default is 'custom_cnn'.
        autoencoder_type : str, optional
            The type of autoencoder to use. Options are 'standard' or
            'variational'.
        number_of_blocks : int, optional
            The number of convolutional blocks to use in the model.
            Default is 4.
        filters : list or int, optional
            The number of filters to use in each block. If a list is provided,
            it must be of length 'number_of_blocks'.
            Default is [16,16,32,32].
        kernel_sizes : list or int, optional
            The kernel size to use in each block. If a list is provided, it must
            be of length 'number_of_blocks'.
            Default is [7,5,3,3].
        strides : list or int, optional
            The stride to use in each block. If a list is provided, it must be
            of length 'number_of_blocks'.
            Default is [2,1,2,1].
        use_maxpool : list or bool, optional
            Whether to use max pooling in each block. If a list is provided, it
            must be of length 'number_of_blocks'.
            Default is False.
        hidden_activation : str, optional
            The activation function to use in the hidden layers. Options are
            'relu', 'selu', 'gelu', or 'linear'.
            Default is 'relu'.
        latent_activation : str, optional
            The activation function to use in the latent layer. Options are
            'relu', 'selu', 'gelu', or 'linear'.
            Default is 'linear'.
        output_activation : str, optional
            The activation function to use in the output layer. Options are
            'linear' or 'sigmoid'.
            Default is 'linear'.
        global_pool_type : str, optional
            The type of global pooling to use. Options are 'avg' or 'max'.
            Default is 'avg'.
        dropout : float, optional
            The dropout rate to use in the model. If None, no dropout is used.
            Default is None.
        use_awgn : bool, optional
            Whether to use additive white Gaussian noise.
            Default is False.
        awgn_variance : float, optional
            The variance of the AWGN.
            Default is 0.1.
        distance : str, optional
            The distance metric to use for the domain learner model.
            Default is 'euclidean'.
        similarity : str, optional
            The similarity metric to use for the domain learner model.
            Default is 'gaussian'.
        similarity_c : float, optional
            The constant to use for the similarity metric.
            Default is 1.0.
        initial_protos : np.ndarray, optional
            The initial prototypes to use for the domain learner model.
            Default is None.
        """
        # check that the data loaders have been created
        assert self.data_loaders_created, \
            "Must create data loaders before creating models."
        
        # do some parameter checking
        assert hidden_activation in ['relu', 'selu', 'gelu', 'linear'], \
            "Hidden activation must be one of 'relu', 'selu', 'gelu', or \
                'linear'."
        assert latent_activation in ['relu', 'selu', 'gelu', 'linear'], \
            "Latent activation must be one of 'relu', 'selu', 'gelu', or \
                'linear'."
        assert output_activation in ['linear', 'sigmoid', 'softmax'], \
            "Output activation must be one of 'linear','sigmoid', or 'softmax'."

        # save off important values as attributes
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.autoencoder_type = autoencoder_type
        self.output_activation = output_activation
        if self.learner_type == 'domain_learner':
            self.distance = distance
            self.similarity = similarity
            self.sim_c = similarity_c

        # save these values in the params dictionary
        self.params['latent_dim'] = latent_dim
        self.params['architecture'] = architecture
        self.params['autoencoder_type'] = autoencoder_type
        self.params['number_of_blocks'] = number_of_blocks
        self.params['filters'] = filters
        self.params['kernel_sizes'] = kernel_sizes
        self.params['strides'] = strides
        self.params['use_maxpool'] = use_maxpool
        self.params['hidden_activation'] = hidden_activation
        self.params['latent_activation'] = latent_activation
        self.params['output_activation'] = output_activation
        self.params['global_pool_type'] = global_pool_type
        self.params['dropout'] = dropout
        self.params['use_awgn'] = use_awgn
        self.params['awgn_variance'] = awgn_variance
        self.params['distance'] = distance
        self.params['similarity'] = similarity
        self.params['similarity_c'] = similarity_c

        # get the appropriate internal method for creating the models
        if self.learner_type == 'classifier':
            model_method = self._create_classifier_learner
        elif self.learner_type == 'autoencoder':
            model_method = self._create_autoencoder_learner
        elif self.learner_type in ['domain_learner']:
            model_method = self._create_semantic_learner
        else:
            raise ValueError("Got invalid learner_type.")
        
        # call the internal method for creating the models
        model_method(
            architecture=architecture,
            autoencoder_type=autoencoder_type,
            latent_dim=latent_dim,
            number_of_blocks=number_of_blocks,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            use_maxpool=use_maxpool,
            hidden_activation=hidden_activation,
            latent_activation=latent_activation,
            output_activation=output_activation,
            global_pool_type=global_pool_type,
            dropout=dropout,
            use_awgn=use_awgn,
            awgn_variance=awgn_variance,
            distance=distance,
            similarity=similarity,
            similarity_c=similarity_c,
            initial_protos=initial_protos
        )

    def compile_learner(
            self,
            loss: Optional[str] = 'mse',
            optimizer: Optional[str] = 'adam',
            learning_rate: Optional[float] = 1e-3,
            weight_decay: Optional[float] = None,
            clipnorm: Optional[float] = None,
            clipvalue: Optional[float] = None,
            metrics: Optional[list] = None,
            alpha: Optional[float] = 1.0,
            beta: Optional[float] = 1.0,
            lam: Optional[float] = 0.01,
            schedule_type: Optional[str] = None,
            sch_init_lr: Optional[float] = 1e-4,
            sch_decay_steps: Optional[int] = 10000,
            sch_warmup_target: Optional[float] = None,
            sch_warmup_steps: Optional[int] = None,
            metric_matrix: Optional[np.ndarray] = None,
            wasserstein_lam: Optional[float] = 1.0,
            wasserstein_p: Optional[float] = 1.0,
            scaled_prior: Optional[bool] = False
        ):
        """
        Method for compiling the models. This method should be called after
        creating the models.

        Parameters
        ----------
        loss : str, optional
            The loss function to use for training.
            Options for 'classifier' are 'categorical_crossentropy', 
                'wasserstein'.
            Options for 'autoencoder' are 'mse' or 'ssim'.
            Options for 'domain_learner' are 'basic' or 'wasserstein'
            Default is 'mse'.
        optimizer : str, optional
            The optimizer to use for training.
            Options for all are 'adam'.
            Default is 'adam'.
        learning_rate : float, optional
            The learning rate to use for training.
            Default is 1e-3.
        weight_decay : float, optional
            The strength of the weight decay to use for training.
            If None, no weight decay is used.
            Default is None.
        clipnorm : float, optional
            If set, the gradient of each weight is individually clipped so that
            its norm is no higher than this value.
            Default is None.
        clipvalue : float, optional
            If set, the gradient of each weight is clipped to be no higher than
            this value.
            Default is None.
        metrics : list, optional
            A list of metrics to use for training.
            Default is None.
        alpha : float, optional
            The weight for the reconstruction loss (domain learner only).
            Default is 1.0.
        beta : float, optional
            The weight for the classification loss (domain learner only).
            Default is 1.0.
        lam : float, optional
            The weight for the semantic distance reg. term (domain learner only)
            Default is 0.01.
        schedule_type : str, optional
            String identifier for the learning rate schedule to use. Options are
            'cosine'.
            Default is None.
        sch_init_lr : float, optional
            The initial learning rate for the schedule.
            Only used if 'schedule' is not None.
            Default is 1e-4.
        sch_decay_steps : int, optional
            The number of steps before decay. Note this is steps (batches), not 
            epochs.
            Only used if 'schedule' is not None.
            Default is 10000.
        sch_warmup_target : float, optional
            The target learning rate for the warmup phase.
            Only used if 'schedule' is 'cosine'.
            Default is 1e-3.
        sch_warmup_steps : int, optional
            The number of steps for the warmup phase.
            Only used if 'schedule' is 'cosine'.
            Default is 1000.
        metric_matrix : np.ndarray, optional
            The matrix of distances between the classes.
            Only used for the Wasserstein loss.
            For the domain_learner, if None, the matrix is dynamically computed
            from the prototypes learned during training.
            Default is None.
        wasserstein_lam : float, optional
            The balancing parameter for the Wasserstein loss.
            Default is 1.0.
        wasserstein_p : float, optional
            The exponent for the distance metric in the Wasserstein loss.
            To get a valid metric, this should be >= 1.
            Default is 1.0.
        scaled_prior : bool, optional
            Whether to use the scaled prior for the VAE.
            Only used for the variational autoencoder/domain learner.
            Default is False.
        """
        # check that the models have been created or loaded
        assert self.models_created or self.models_loaded, \
            "Must create or load models before compiling them."
            
        # save off important values as attributes
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

        # save these values in the params dictionary
        self.params['loss'] = loss
        self.params['optimizer'] = optimizer
        self.params['learning_rate'] = learning_rate
        self.params['alpha'] = alpha
        self.params['beta'] = beta
        self.params['lam'] = lam

        # get the appropriate internal method for compiling the models
        if self.learner_type == 'classifier':
            compile_method = self._compile_classifier_learner
        elif self.learner_type == 'autoencoder':
            compile_method = self._compile_autoencoder_learner
        elif self.learner_type in ['domain_learner']:
            compile_method = self._compile_semantic_learner
        else:
            raise ValueError("Got invalid learner type.")
        
        # if a learning rate schedule is specified, create the schedule object
        if schedule_type == 'cosine':
            schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=sch_init_lr,
                decay_steps=sch_decay_steps,
                warmup_target=sch_warmup_target,
                warmup_steps=sch_warmup_steps
            )
        elif schedule_type is None:
            schedule = None
        else:
            raise ValueError("Got invalid learning rate schedule type.")
        
        # call the internal method for compiling the models
        compile_method(
            loss=loss,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            metrics=metrics,
            alpha=alpha,
            beta=beta,
            lam=lam,
            schedule=schedule,
            metric_matrix=metric_matrix,
            wasserstein_lam=wasserstein_lam,
            wasserstein_p=wasserstein_p,
            scaled_prior=scaled_prior
        )

        # set flag to indicate that models have been compiled
        self.models_compiled = True

    def load_pretrained_learner(
            self,
            learner_path: str,
        ):
        """
        Method for loading in pretrained models for testing/more training.

        Parameters
        ----------
        learner_path : str
            The path to the directory containing the saved models.
        """
        # make sure the path exists
        assert os.path.exists(learner_path), \
            "The provided path does not exist."
        
        # make sure path ends with a /
        if learner_path[-1] != '/':
            learner_path += '/'
        
        # get the files in the learner directory
        files = os.listdir(learner_path)

        # load in the model parameters
        if 'params.json' in files:
            with open(learner_path + 'params.json', 'r') as f:
                self.params = json.load(f)
        else:
            raise ValueError("Could not find the params.json file.")
        
        # if a domain learner, load in the prototypes
        if self.learner_type in ['domain_learner']:
            if 'prototypes.npy' in files:
                self.prototypes = np.load(learner_path + 'prototypes.npy')
            else:
                self.prototypes = np.random.normal(
                    size=(
                        self.number_of_properties,
                        self.params['latent_dim']
                    )
                )
        else:
            self.prototypes = None

        # create the model architecture based on the params
        self.create_learner(
            latent_dim=self.params['latent_dim'],
            architecture=self.params['architecture'],
            autoencoder_type=self.params['autoencoder_type'],
            number_of_blocks=self.params['number_of_blocks'],
            filters=self.params['filters'],
            kernel_sizes=self.params['kernel_sizes'],
            strides=self.params['strides'],
            use_maxpool=self.params['use_maxpool'],
            hidden_activation=self.params['hidden_activation'],
            latent_activation=self.params['latent_activation'],
            output_activation=self.params['output_activation'],
            global_pool_type=self.params['global_pool_type'],
            dropout=self.params['dropout'],
            use_awgn=self.params['use_awgn'],
            awgn_variance=self.params['awgn_variance'],
            distance=self.params['distance'],
            similarity=self.params['similarity'],
            similarity_c=self.params['similarity_c'],
            initial_protos=self.prototypes,
        )

        # make sure the model is built before loading in weights
        for data, labels in self.training_loader.take(1):
            data = data.numpy()
            labels = labels[1].numpy()
        self.model(data)
        
        # if the learner type is the same, load in the models
        if self.params['learner_type'] == self.learner_type:
            self.model.load_weights(learner_path + 'model.weights.h5')
        # else, only load in the encoder/decoder weights as needed
        else:
            self.encoder.load_weights(learner_path + 'encoder.weights.h5')
            if self.decoder_created:
                self.decoder.load_weights(learner_path + 'decoder.weights.h5')

        # set flag to indicate that models have been loaded
        self.models_loaded = True
        self.encoder_loaded = True if self.encoder_created else False
        self.decoder_loaded = True if self.decoder_created else False

    def train_learner(
            self,
            epochs: Optional[int] = 5,
            steps_per_epoch: Optional[int] = None,
            validation_steps: Optional[int] = None,
            callbacks: Optional[list] = None,
            verbose: Optional[int] = 1,
            proto_update_type: Optional[str] = 'average',
            proto_update_step_size: Optional[int] = None,
            mu: Optional[float] = 0.5,
            warmup: Optional[int] = 0,
            log_experiment: Optional[bool] = False,
            proto_plot_save_path: Optional[str] = None,
            proto_plot_colors: Optional[list] = None,
            proto_plot_legend: Optional[list] = None,
            fixed_prototypes: Optional[bool] = False
        ):
        """
        Method for training the learner. This method should be called after
        compiling the models.

        Parameters
        ----------
        epochs : int, optional
            The number of epochs to train for.
            Default is 5.
        steps_per_epoch : int, optional
            The number of batches trained on per epoch.
            Default is None.
        validation_steps : int, optional
            The number of batches validated on per epoch.
            Default is None.
        callbacks : list, optional
            A list of tf.keras.callbacks.Callback objects to use for training.
            These will be passed directly to the model.fit method.
            Default is None.
        verbose : int, optional
            The verbosity level to use for training. 1 is progress bar, 2 is
            one line per epoch, 0 is silent.
            Default is 1.
        proto_update_type : str, optional
            The type of prototype update to use. Options are 'average'.
            Only applies to domain learning.
            Default is 'average'.
        proto_update_step_size : int, optional
            The number of batches used to update the prototypes.
            Only applies to domain learning.
            Default is None.
        mu : float, optional
            The "mixing parameter" for the prototype update.
            1.0 just uses the old prototype, 0.0 is full update.
            Only applies to domain learning.
            Default is 0.5.
        warmup : int, optional
            The number of epochs to train without semantic regularization.
            Only applies to domain learning.
            Default is 0.
        log_experiment : bool, optional
            Whether to log the experiment to comet ML.
            Default is False.
        proto_plot_save_path : str, optional
            Path to save the prototype plots. If None, the plots are not saved.
            Only applies to domain learning.
            Default is None.
        proto_plot_colors : list, optional
            A list of colors to use for the prototype plots, as hex strings.
            Only applies to domain learning.
            Default is None.
        proto_plot_legend : list, optional
            A list of strings to use for the prototype plots legend.
            Only applies to domain learning.
            Default is None.
        fix_prototypes : bool, optional
            Whether to fix the prototypes during training.
            Only applies to domain learning.
            Default is False.
        """

        # check that the models have been compiled
        assert self.models_compiled, \
            "Must compile models before training them. To compile the models, \
                use the 'compile_learner' method."
        
        # do some parameter checking
        assert verbose in [0, 1, 2], \
            "Verbose must be one of 0, 1, or 2."
        
        # compute the number of steps per epoch if not provided
        if steps_per_epoch is None:
            if self.train_size % self.batch_size == 0:
                steps_per_epoch = self.train_size // self.batch_size
            else:
                steps_per_epoch = self.train_size // self.batch_size + 1
        if validation_steps is None:
            if self.valid_size % self.batch_size == 0:
                validation_steps = self.valid_size // self.batch_size
            else:
                validation_steps = self.valid_size // self.batch_size + 1

        # save off important values as attributes
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.callbacks = callbacks

        # save these values in the params dictionary
        self.params['epochs'] = epochs
        self.params['steps_per_epoch'] = steps_per_epoch
        self.params['validation_steps'] = validation_steps
        
        # get the appropriate internal method for training the models
        if self.learner_type == 'classifier':
            train_method = self._train_classifier_model
        elif self.learner_type == 'autoencoder':
            train_method = self._train_autoencoder_model
        elif self.learner_type in ['domain_learner']:
            train_method = self._train_semantic_model
        else:
            raise ValueError("Got invalid learner type.")
        
        # call the internal method for setting up the comet experiment
        training_context = self._setup_comet_experiment(log_experiment)

        # call the internal method for training the models
        with training_context:
            train_method(
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=verbose,
                proto_update_type=proto_update_type,
                proto_update_step_size=proto_update_step_size,
                mu=mu,
                warmup=warmup,
                proto_plot_save_path=proto_plot_save_path,
                proto_plot_colors=proto_plot_colors,
                proto_plot_legend=proto_plot_legend,
                fixed_prototypes=fixed_prototypes
            )

    def save_models(
          self,
          save_path: str,  
        ):
        """
        Method to save the models after training.

        Parameters
        ----------
        save_path : str
            The path to the directory to save the models to.
        """
        # check that the models have been trained
        assert self.models_trained, \
            "Models have not been trained. To train the models, use the \
                'train_learner' method."
        
        # make sure path ends with a /
        if save_path[-1] != '/':
            save_path += '/'

        # create a directory with the data set, learner type, and timestamp
        save_path += self.dataset + '_' + self.learner_type + '_' + \
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'
        os.mkdir(save_path)
        
        # save the parameter dictionary
        with open(save_path + 'params.json', 'w') as f:
            json.dump(self.params, f)

        # save the model weights
        self.model.save_weights(save_path + 'model.weights.h5')
        if self.encoder is not None:
            self.encoder.save_weights(save_path + 'encoder.weights.h5')
        if self.decoder is not None:
            self.decoder.save_weights(save_path + 'decoder.weights.h5')

        # save the prototypes
        if self.learner_type == 'domain_learner':
            np.save(save_path + 'prototypes.npy', self.prototypes)
        
        print(f'Models saved to {save_path}.')

    def summarize_models(self):
        """
        Method for printing a summary of the created models.
        """
        # check that the models have been created or loaded
        assert self.models_created or self.encoder_loaded or \
            self.decoder_loaded or self.models_loaded, \
            "Models have not been created or loaded. To create the models, use " \
                + "the 'create_learner' method. To load the models, use the " \
                + "'load_pretrained_learner' method."
        
        data_shape = (None, self.height, self.width, self.channels)
        latent_shape = (None, self.latent_dim)

        # print the summary of the encoder
        print('\n\n')
        utils.print_model_summary(model=self.encoder, input_shape=data_shape)

        # if the decoder has been created, print the summary
        if self.decoder_created or self.decoder_loaded:
            print('\n\n')
            utils.print_model_summary(
                model=self.decoder, 
                input_shape=latent_shape
            )

        # print the summary of the overall model
        print('\n\n')
        utils.print_model_summary(model=self.model, input_shape=data_shape)

    def eval_plot_loss_curves(
            self,
            which: Optional[str] = 'both',
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        Method for plotting the training loss.

        Parameters
        ----------
        which : str, optional
            Which loss curves to plot. Options are 'training', 'validation',
            or 'both'.
            Default is 'both'.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        """
        # check that the models have been trained
        if not self.models_trained:
            print("Models have not been trained. Skipping...")
            return
    
        # do some parameter checking
        if which not in ['training', 'validation', 'both']:
            print("which must be one of 'training', 'validation', or 'both'."+ \
                " Skipping...")
            return
        
        # pull out the losses from the training history
        trn_loss = self.training_history['loss']
        vld_loss = self.training_history['val_loss']
        epochs = np.arange(1, len(trn_loss)+1)

        # plot the losses
        plt.figure(figsize=(10, 5))
        if which == 'training' or which == 'both':
            plt.plot(epochs, trn_loss, label='Training Loss')
        if which == 'validation' or which == 'both':
            plt.plot(epochs, vld_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        # save the plot with an unused file name if a save path is provided
        if save_path is not None:
            vacant_save_path = utils.get_unused_name(save_path)
            plt.savefig(vacant_save_path)

        # show the plot
        if show:
            plt.show(block=block)

    def eval_plot_accuracy_curves(
            self,
            which: Optional[str] = 'both',
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        Method for plotting the training accuracy.

        Parameters
        ----------
        which : str, optional
            Which accuracy curves to plot. Options are 'training', 'validation',
            or 'both'.
            Default is 'both'.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
        """
        # check that the models have been trained
        if not self.models_trained:
            print("Models have not been trained. Skipping...")
            return
        
        # do some parameter checking
        if which not in ['training', 'validation', 'both']:
            print("which must be one of 'training', 'validation', or 'both'."+ \
                " Skipping...")
            return
        
        # check that accuracy is available
        if 'accuracy' not in self.training_history.keys():
            print("Accuracy is not available for this model. Skipping...")
            return

        # pull out the losses from the training history
        trn_acc = self.training_history['accuracy']
        vld_acc = self.training_history['val_accuracy']
        epochs = np.arange(1, len(trn_acc)+1)

        # plot the losses
        plt.figure(figsize=(10, 5))
        if which == 'training' or which == 'both':
            plt.plot(epochs, trn_acc, label='Training Accuracy')
        if which == 'validation' or which == 'both':
            plt.plot(epochs, vld_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        # save the plot with an unused file name if a save path is provided
        if save_path is not None:
            vacant_save_path = utils.get_unused_name(save_path)
            plt.savefig(vacant_save_path)

        # show the plot
        if show:
            plt.show(block=block)

    def eval_compare_latent_prior(
            self,
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        Method for comparing the latent space prior to the learned latent space.
        """
        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        loader = self.eval_loader
        set_size = self.valid_size
        
        # get the prior distribution
        if self.model.scaled_prior:
            var = 1.0/self.latent_dim
            std = np.sqrt(var)
        else:
            var = 1.0
            std = 1.0
        x_lim = 5*std
        x = np.linspace(-x_lim, x_lim, 250)
        f = (1/np.sqrt(2*np.pi*var)) * np.exp(-0.5*(x**2)/var)

        # get the learned latent features
        features = np.empty((0, self.latent_dim))
        total = 0
        for batch_data, batch_labels in loader:
            batch_features = self.encoder.predict(batch_data, verbose=0)
            if self.autoencoder_type == 'variational':
                    batch_features = layers.ReparameterizationLayer(
                        latent_dim=self.latent_dim
                    )(batch_features)[2]
            features = np.append(features, batch_features, axis=0)
            total += len(batch_labels)
            print(f'\rComputing features ({total}/{set_size})', end='')
            if total >= set_size:
                break
        print('\nDone.')

        # clip the features to the same range as the prior
        features = np.clip(features, -x_lim, x_lim)

        # plot the prior density and histogram of the learned latent features
        plt.figure(figsize=(10, 5))
        plt.plot(x, f, label='Prior')
        plt.hist(
            features.flatten(), 
            bins=50, 
            density=True, 
            alpha=0.5, 
            label='Learned'
        )

        # set the plot labels
        plt.xlabel('Latent Feature Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()

        # save the plot with an unused file name if a save path is provided
        if save_path is not None:
            vacant_save_path = utils.get_unused_name(save_path)
            plt.savefig(vacant_save_path)

        # show the plot
        if show:
            plt.show(block=block)

    def eval_plot_scattered_features(
            self,
            which: Optional[str] = 'validation',
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True,
            colors: Optional[list] = None,
            legend: Optional[list] = None
        ):
        """
        Method for plotting the scattered features that are learned by the
        encoder.

        Parameters
        ----------
        which : str, optional
            Which features to plot. Options are 'training', 'validation'.
            Default is 'validation'.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        colors : list, optional
            A list of colors to use for the different classes.
            Default is None, for which colors are automatically chosen.
        legend : list, optional
            A list of strings to use for the legend.
            Default is None, for which no legend is shown
        """
        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        
        # do some parameter checking
        if which not in ['training', 'validation']:
            print("which must be one of 'training' or 'validation'. Skipping...")
            return
        if colors is not None:
            if not isinstance(colors, list):
                print("colors must be a list. Skipping...")
                return
        if legend is not None:
            if not isinstance(legend, list):
                print("legend must be a list. Skipping...")
                return
            
        # get the appropriate data loader
        if which == 'training':
            print('Training data not yet supported here. Skipping...')
            return
        elif which == 'validation':
            loader = self.eval_loader
            set_size = self.valid_size

        # get the features and labels from the data loader
        if self.latent_dim == 2:
            features = np.empty((0,2))
        elif self.latent_dim == 3:
            features = np.empty((0,3))
        else:
            print("Features must be either 2D or 3D. Skipping...")
            return
        labels = np.empty((0, self.number_of_properties))
        total = 0
        for batch_data, batch_labels in loader:
            batch_features = self.encoder.predict(batch_data, verbose=0)
            if self.autoencoder_type == 'variational':
                    batch_features = layers.ReparameterizationLayer(
                        latent_dim=self.latent_dim
                    )(batch_features)[2]
            features = np.append(features, batch_features, axis=0)
            labels = np.append(labels, batch_labels, axis=0)
            total += len(batch_labels)
            print(f'\rComputing features ({total}/{set_size})', end='')
            if total >= set_size:
                break
        print('\nDone.')

        # pass to the method for plotting the features
        vis.plot_scattered_features(
            features=features,
            labels=labels,
            show=show,
            save_path=save_path,
            block=block,
            colors=colors,
            legend=legend
        )
    
    def eval_show_decoded_protos(
            self,
            legend: Optional[list] = None,
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        This function shows the images resulting from the decoded prototypes.
        Only applicable to the domain learner model.

        Parameters
        ----------
        legend : list, optional
            A list of strings to use for the legend.
            Default is None, for which no legend is shown.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        """
        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        # check that the model is a domain learner
        if self.learner_type not in ['domain_learner']:
            print("This method is only applicable to the domain learner. " + \
                "Skipping...")
            return
        
        # call the method for decoding and showing the prototypes
        vis.plot_decoded_protos(
            decoder=self.decoder,
            prototypes=self.prototypes,
            dataset=self.dataset,
            legend=legend,
            show=show,
            save_path=save_path,
            block=block
        )

    def eval_plot_scattered_protos(
            self,
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True,
            colors: Optional[list] = None,
            legend: Optional[list] = None
        ):
        """
        Method for plotting the scattered prototypes that are learned by the
        domain learner.

        Parameters
        ----------
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        colors : list, optional
            A list of colors to use for the different classes.
            Default is None, for which colors are automatically chosen.
        legend : list, optional
            A list of strings to use for the legend.
            Default is None, for which no legend is shown
        """
        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        # make sure features are either 2D or 3D
        if self.latent_dim not in [2, 3]:
            print("Features must be either 2D or 3D. Skipping...")
            return
        
        # check that the model is a domain learner
        if self.learner_type not in ['domain_learner']:
            print("This method is only applicable to the domain learner. " + \
                "Skipping...")
            return
        
        # do some parameter checking
        if colors is not None:
            if not isinstance(colors, list):
                print("Colors must be a list. Skipping...")
                return
        if legend is not None:
            if not isinstance(legend, list):
                print("Legend must be a list. Skipping...")
                return
            
        # call the method for plotting the scattered prototypes
        vis.plot_scattered_prototypes(
            prototypes=self.prototypes,
            show=show,
            save_path=save_path,
            block=block,
            colors=colors,
            legend=legend
        )

    def eval_compare_true_and_generated(
            self,
            which: Optional[str] = 'validation',
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        Method for comparing the true data and the generated data.

        Parameters
        ----------
        which : str, optional
            Which data to compare. Options are 'training' or 'validation'.
            Default is 'validation'.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        """

        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        # check that the model is an autoencoder or domain learner
        if self.learner_type != 'autoencoder' and \
            self.learner_type not in ['domain_learner']:
            print("This method is only applicable to autoencoder/domain learners. " + \
                "Skipping...")
            return
        
        # get some data from the specified loader
        if which == 'training':
            batch = self.training_loader.take(1)
        elif which == 'validation':
            batch = self.validation_loader.take(1)
        else:
            print("Got invalid 'which' parameter. Skipping...")
            return
        for data, labels in batch:
            if self.learner_type == 'autoencoder':
                data = data.numpy()
                labels = labels.numpy()
            elif self.learner_type in ['domain_learner']:
                data = data.numpy()
                labels = labels[0].numpy()
            else:
                print("Got invalid learner type. Skipping...")
                return

        # get whether the model is a VAE or not
        is_vae = self.autoencoder_type == 'variational'

        # call the method for comparing the true and generated data
        vis.plot_true_and_decoded(
            data=data,
            labels=labels,
            encoder=self.encoder,
            decoder=self.decoder,
            dataset=self.dataset,
            is_variational=is_vae,
            show=show,
            save_path=save_path,
            block=block
        )

    def eval_plot_similarity_heatmap(
            self,
            legend,
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        Method for plotting the similarity heatmap for the domain learner.

        Parameters
        ----------
        legend : list
            A list of strings to use for the legend.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        """

        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        # check that the model is a domain learner
        if not self.learner_type in ['domain_learner']:
            print("This method is only applicable to the domain learner. " + \
                "Skipping...")
            return
        
        # compute the similarity matrix
        D = np.zeros((self.number_of_properties, self.number_of_properties))
        for i, proto in enumerate(self.prototypes):
            for j, proto2 in enumerate(self.prototypes):
                D[i,j] = np.linalg.norm(proto - proto2)
        S = np.exp(-self.sim_c*D**2)

        # plot the similarity heatmap
        vis.plot_heatmap(
            matrix=S,
            title='Similarity Matrix',
            legend=legend,
            save_path=save_path,
            show=show,
            block=block
        )

    def eval_visualize_dimension(
            self,
            dimension: int,
            min_val: Optional[float] = None,
            max_val: Optional[float] = None,
            steps: Optional[int] = 10,
            fixed_dims: Optional[list] = None,
            is_random_fixed_dims: Optional[bool] = False,
            is_grayscale: Optional[bool] = False,
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        Method for showing the effect of a single dimension on the generated
        images.

        Parameters
        ----------
        dimension : int
            The dimension to visualize.
        min_val : float, optional
            The minimum value for the dimension.
            If None, the minimum value in the data is used.
            Default is None.
        max_val : float, optional
            The maximum value for the dimension.
            If None, the maximum value in the data is used.
            Default is None.
        steps : int, optional
            The number of steps to take between the min and max values.
            Default is 10.
        fixed_dims : list, optional
            A list of fixed values for the other dimensions.
            If None, the other dimensions are fixed at 0.
            Default is None.
        is_random_fixed_dims : bool, optional
            Whether to randomly choose the fixed dimensions.
            Default is False.
        is_grayscale : bool, optional
            Whether to show the images in grayscale.
            Default is False.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        """
        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        # check that the model is a domain learner or autoencoder
        if not self.learner_type in ['domain_learner'] and \
            self.learner_type != 'autoencoder':
            print("This method is only applicable to the domain" + \
                   " learner and autoencoder. Skipping...")
            return
        
        # get the encoded features
        features = np.empty((0, self.latent_dim))
        set_size = self.valid_size
        total = 0
        for batch_data, _ in self.validation_loader:
            batch_features = self.encoder.predict(batch_data, verbose=0)
            if self.autoencoder_type == 'variational':
                batch_features = layers.ReparameterizationLayer(
                    latent_dim=self.latent_dim
                )(batch_features)[2]
            features = np.append(features, batch_features, axis=0)
            total += len(batch_data)
            print(f'\rComputing features ({total}/{set_size})', end='')
            if total >= set_size:
                break
        print('\nDone.')

        # get min and max vals if not entered
        if min_val is None:
            min_val = np.min(features[:,dimension])
        if max_val is None:
            max_val = np.max(features[:,dimension])
        
        vis.visualize_dimension(
            features=features,
            decoder=self.decoder,
            dim=dimension,
            min_val=min_val,
            max_val=max_val,
            steps=steps,
            fixed_dims=fixed_dims,
            is_random_fixed_dims=is_random_fixed_dims,
            is_grayscale=is_grayscale,
            show=show,
            save_path=save_path,
            block=block
        )

    def eval_visualize_all_dimensions(
            self,
            steps: Optional[int] = 10,
            fixed_dims: Optional[list] = None,
            is_random_fixed_dims: Optional[bool] = False,
            is_grayscale: Optional[bool] = False,
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        Method for showing the effect of all dimensions on the generated
        images.

        Parameters
        ----------
        steps : int, optional
            The number of steps to take between the min and max values.
            Default is 10.
        fixed_dims : list, optional
            A list of dimensions to fix.
            If specified, list should be of length n_features.
            If not specified, all other dimensions are either fixed at their
            respective means, or are randomly chosen if is_random_fixed_dims
            is True.
        Default value is None.
        is_random_fixed_dims : bool, optional
            Whether to randomly choose the fixed dimensions.
            Default is False.
        is_grayscale : bool, optional
            Whether to show the images in grayscale.
            Default is False.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        """
        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        # check that the model is a domain learner or autoencoder
        if not self.learner_type in ['domain_learner'] and \
            self.learner_type != 'autoencoder':
            print("This method is only applicable to the domain" + \
                   " learner and autoencoder. Skipping...")
            return
        
        # get the encoded features
        features = np.empty((0, self.latent_dim))
        set_size = self.valid_size
        total = 0
        for batch_data, _ in self.validation_loader:
            batch_features = self.encoder.predict(batch_data, verbose=0)
            if self.autoencoder_type == 'variational':
                batch_features = layers.ReparameterizationLayer(
                    latent_dim=self.latent_dim
                )(batch_features)[2]
            features = np.append(features, batch_features, axis=0)
            total += len(batch_data)
            print(f'\rComputing features ({total}/{set_size})', end='')
            if total >= set_size:
                break
        print('\nDone.')

        vis.visualize_all_dimensions(
            features=features,
            decoder=self.decoder,
            steps=steps,
            fixed_dims=fixed_dims,
            is_random_fixed_dims=is_random_fixed_dims,
            is_grayscale=is_grayscale,
            show=show,
            save_path=save_path,
            block=block
        )

    def eval_plot_similarity_histograms(
            self,
            legend: list,
            show: Optional[bool] = True,
            save_path: Optional[str] = None,
            block: Optional[bool] = True
        ):
        """
        Method for plotting the similarity histograms for the domain learner.

        Parameters
        ----------
        legend : list
            A list of strings to use for the legend.
        show : bool, optional
            Whether to show the plot.
            Default is True.
        save_path : str, optional
            The path to save the plot to. If None, the plot is not saved.
            Default is None.
        block : bool, optional
            Whether to block the execution until the plot window is closed.
            Default is True.
        """
        # check that the models have been trained or loaded
        if not self.models_trained and not self.models_loaded:
            print("Models have not been trained or loaded. Skipping...")
            return
        
        # check that the model is a domain learner
        if not self.learner_type in ['domain_learner']:
            print("This method is only applicable to the domain learner. " + \
                "Skipping...")
            return
        
        set_size = self.valid_size
        
        # get the encoded features
        print('Computing features...')
        features = np.empty((0, self.latent_dim))
        labels = np.empty((0, self.number_of_properties))
        i = 0
        total = 0
        for batch_data, batch_labels in self.validation_loader:
            batch_features = self.encoder.predict(batch_data, verbose=0)
            if self.autoencoder_type == 'variational':
                batch_features = layers.ReparameterizationLayer(
                    latent_dim=self.latent_dim
                )(batch_features)[2]
            labels = np.append(labels, batch_labels[1], axis=0)
            features = np.append(features, batch_features, axis=0)
            print(i)
            total += len(batch_labels)
            if total >= set_size:
                break
            i += 1

        # call the method for plotting the similarity histograms
        vis.plot_similarity_histograms(
            features=features,
            labels=labels,
            number_of_properties=self.number_of_properties,
            legend=legend,
            show=show,
            save_path=save_path,
            block=block,
            similarity_c=self.sim_c
        )
    
    # ----------------------- internal methods ---------------------------------
    # v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v 

    def _create_classifier_data_loaders(
            self,
            dataset: str,
            batch_size: Optional[int] = 32,
            buffer_size: Optional[int] = 10000,
            paths_dict: Optional[dict] = None,
            shapes_dict: Optional[dict] = None
        ):
        """
        Internal method for creating the data loaders for the classifier model.
        Parameters are the same as the create_data_loaders method.

        This method sets the following attributes:
        - training_loader : tf.Data.Dataset object for the training data
        - validation_loader : tf.Data.Dataset object for the validation data
        """
        if dataset == 'mnist':
            ((x_trn, y_trn), (x_vld, y_vld)) = \
                tf.keras.datasets.mnist.load_data(path='mnist.npz')
        elif dataset == 'cifar10':
            ((x_trn, y_trn), (x_vld, y_vld)) = \
                tf.keras.datasets.cifar10.load_data()
        else:
            ((x_trn, y_trn), (x_vld, y_vld)) = ((None, None), (None, None))
            
        # if data is loaded from tf pre-defined datasets
        if x_trn is not None:
            trn_loader = feeders.ClassifierFeederFromArray(
                is_training=True,
                data_array=x_trn,
                label_array=y_trn,
                batch_size=batch_size
            )
            vld_loader = feeders.ClassifierFeederFromArray(
                is_training=False,
                data_array=x_vld,
                label_array=y_vld,
                batch_size=batch_size
            )
            eval_loader = feeders.ClassifierFeederFromArray(
                is_training=False,
                data_array=x_vld,
                label_array=y_vld,
                batch_size=batch_size
            )
        # otherwise, load in the data from the provided paths
        else:
            trn_loader = feeders.ClassifierFeederFromMemmap(
                is_training=True,
                data_file=paths_dict['train_data_path'],
                label_file=paths_dict['train_labels_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['train_data_shape'],
                label_array_shape=shapes_dict['train_labels_shape']
            )
            vld_loader = feeders.ClassifierFeederFromMemmap(
                is_training=False,
                data_file=paths_dict['valid_data_path'],
                label_file=paths_dict['valid_labels_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['valid_data_shape'],
                label_array_shape=shapes_dict['valid_labels_shape']
            )
            eval_loader = feeders.ClassifierFeederFromMemmap(
                is_training=False,
                data_file=paths_dict['valid_data_path'],
                label_file=paths_dict['valid_labels_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['valid_data_shape'],
                label_array_shape=shapes_dict['valid_labels_shape']
            )

        # save the data loaders as attributes of the class
        self.training_loader = trn_loader
        self.validation_loader = vld_loader
        self.eval_loader = eval_loader

    def _create_autoencoder_data_loaders(
            self,
            dataset: str,
            batch_size: Optional[int] = 32,
            buffer_size: Optional[int] = 10000,
            paths_dict: Optional[dict] = None,
            shapes_dict: Optional[dict] = None
        ):
        """
        Internal method for creating the data loaders for the autoencoder model.
        Parameters are the same as the create_data_loaders method.
        """
        if dataset == 'mnist':
            ((x_trn, _), (x_vld, y_vld)) = \
                tf.keras.datasets.mnist.load_data(path='mnist.npz')
        elif dataset == 'cifar10':
            ((x_trn, _), (x_vld, y_vld)) = \
                tf.keras.datasets.cifar10.load_data()
        else:
            ((x_trn, _), (x_vld, y_vld)) = ((None, None), (None, None))

        # if data is loaded from tf pre-defined datasets
        if x_trn is not None:
            trn_loader = feeders.AutoencoderFeederFromArray(
                is_training=True,
                data_array=x_trn,
                batch_size=batch_size
            )
            vld_loader = feeders.AutoencoderFeederFromArray(
                is_training=False,
                data_array=x_vld,
                batch_size=batch_size
            )
            eval_loader = feeders.ClassifierFeederFromArray(
                is_training=False,
                data_array=x_vld,
                label_array=y_vld,
                batch_size=batch_size
            )

        # otherwise, load in the data from the provided paths
        else:
            trn_loader = feeders.AutoencoderFeederFromMemmap(
                is_training=True,
                data_file=paths_dict['train_data_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['train_data_shape']
            )
            vld_loader = feeders.AutoencoderFeederFromMemmap(
                is_training=False,
                data_file=paths_dict['valid_data_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['valid_data_shape']
            )
            eval_loader = feeders.ClassifierFeederFromMemmap(
                is_training=False,
                data_file=paths_dict['valid_data_path'],
                label_file=paths_dict['valid_labels_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['valid_data_shape'],
                label_array_shape=shapes_dict['valid_labels_shape']
            )

        # save the data loaders as attributes of the class
        self.training_loader = trn_loader
        self.validation_loader = vld_loader
        self.eval_loader = eval_loader
            
    def _create_semantic_data_loaders(
            self,
            dataset: str,
            batch_size: Optional[int] = 32,
            buffer_size: Optional[int] = 10000,
            paths_dict: Optional[dict] = None,
            shapes_dict: Optional[dict] = None
        ):
        """
        Internal method for creating the data loaders for the domain learner
        model. Parameters are the same as the create_data_loaders method.
        """
        if dataset == 'mnist':
            ((x_trn, y_trn), (x_vld, y_vld)) = \
                tf.keras.datasets.mnist.load_data(path='mnist.npz')
        elif dataset == 'cifar10':
            ((x_trn, y_trn), (x_vld, y_vld)) = \
                tf.keras.datasets.cifar10.load_data()
        else:
            ((x_trn, y_trn), (x_vld, y_vld)) = ((None, None), (None, None))

        # if data is loaded from tf pre-defined datasets
        if x_trn is not None:
            trn_loader = feeders.DomainLearnerFeederFromArray(
                is_training=True,
                data_array=x_trn,
                label_array=y_trn,
                batch_size=batch_size
            )
            vld_loader = feeders.DomainLearnerFeederFromArray(
                is_training=False,
                data_array=x_vld,
                label_array=y_vld,
                batch_size=batch_size
            )
            eval_loader = feeders.ClassifierFeederFromArray(
                is_training=False,
                data_array=x_vld,
                label_array=y_vld,
                batch_size=batch_size
            )
        # otherwise, load in the data from the provided paths
        else:
            trn_loader = feeders.DomainLearnerFeederFromMemmap(
                is_training=True,
                data_file=paths_dict['train_data_path'],
                label_file=paths_dict['train_labels_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['train_data_shape'],
                label_array_shape=shapes_dict['train_labels_shape']
            )
            vld_loader = feeders.DomainLearnerFeederFromMemmap(
                is_training=False,
                data_file=paths_dict['valid_data_path'],
                label_file=paths_dict['valid_labels_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['valid_data_shape'],
                label_array_shape=shapes_dict['valid_labels_shape']
            )
            eval_loader = feeders.ClassifierFeederFromMemmap(
                is_training=False,
                data_file=paths_dict['valid_data_path'],
                label_file=paths_dict['valid_labels_path'],
                batch_size=batch_size,
                buffer_size=buffer_size,
                data_array_shape=shapes_dict['valid_data_shape'],
                label_array_shape=shapes_dict['valid_labels_shape']
            )

        # save the data loaders as attributes of the class
        self.training_loader = trn_loader
        self.validation_loader = vld_loader
        self.eval_loader = eval_loader


    def _create_classifier_learner(
            self,
            architecture: str,
            latent_dim: int,
            number_of_blocks: int,
            filters: Union[list,int],
            kernel_sizes: Union[list,int],
            strides: Union[list,int],
            use_maxpool: Union[list,bool],
            hidden_activation: str,
            latent_activation: str,
            output_activation: str,
            global_pool_type: str,
            dropout: float,
            use_awgn: bool,
            awgn_variance: float,
            **kwargs
        ):

        # create the encoder
        encoder = self._create_encoder_model(
            architecture=architecture,
            latent_dim=latent_dim,
            number_of_blocks=number_of_blocks,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            use_maxpool=use_maxpool,
            hidden_activation=hidden_activation,
            latent_activation=latent_activation,
            global_pool_type=global_pool_type,
            dropout=dropout
        )

        classifier = models.Classifier(
            num_classes=self.number_of_properties,
            encoder=encoder,
            output_activation=output_activation,
            use_awgn=use_awgn,
            awgn_variance=awgn_variance,
            name='classifier'
        )

        # set the models as attributes of the class
        self.encoder = encoder
        self.model = classifier

        # build the models
        self.encoder.build(
            input_shape=(None, self.height, self.width, self.channels)
        )
        self.model.build(
            input_shape=(None, self.height, self.width, self.channels)
        )

        # set flags to indicate that the models have been created
        self.encoder_created = True
        self.models_created = True
    
    def _create_autoencoder_learner(
            self,
            architecture: str,
            autoencoder_type: str,
            latent_dim: int,
            number_of_blocks: int,
            filters: Union[list,int],
            kernel_sizes: Union[list,int],
            strides: Union[list,int],
            use_maxpool: Union[list,bool],
            hidden_activation: str,
            latent_activation: str,
            output_activation: str,
            global_pool_type: str,
            dropout: float,
            use_awgn: bool,
            awgn_variance: float,
            **kwargs
        ):

        ae = autoencoder_type
        
        # create the encoder
        encoder = self._create_encoder_model(
            architecture=architecture,
            latent_dim=latent_dim if ae == 'standard' else 2*latent_dim,
            number_of_blocks=number_of_blocks,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            use_maxpool=use_maxpool,
            hidden_activation=hidden_activation,
            latent_activation=latent_activation,
            global_pool_type=global_pool_type,
            dropout=dropout
        )

        # create the decoder
        decoder = self._create_decoder_model(
            encoder=encoder,
            architecture=architecture,
            latent_dim=latent_dim,
            channels=self.channels,
            number_of_blocks=number_of_blocks,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            use_maxpool=use_maxpool,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            dropout=dropout
        )

        # create the autoencoder
        if autoencoder_type == 'standard':
            model = models.Autoencoder(
                encoder=encoder,
                decoder=decoder,
                use_awgn=use_awgn,
                awgn_variance=awgn_variance,
                name='autoencoder'
            )
        elif autoencoder_type == 'variational':
            model = models.VariationalAutoencoder(
                encoder=encoder,
                decoder=decoder,
                latent_dimension=latent_dim,
                input_shape=(self.height, self.width, self.channels),
                use_awgn=use_awgn,
                awgn_variance=awgn_variance,
                name='autoencoder'
            )
        else:
            raise ValueError("Invalid autoencoder type.")

        # set the models as attributes of the class
        self.encoder = encoder
        self.decoder = decoder
        self.model = model

        # build the models
        self.encoder.build(
            input_shape=(None, self.height, self.width, self.channels)
        )
        self.decoder.build(
            input_shape=(None, latent_dim)
        )
        self.model.build(
            input_shape=(None, self.height, self.width, self.channels)
        )

        # set flags to indicate that the models have been created
        self.encoder_created = True
        self.decoder_created = True
        self.models_created = True

    def _create_semantic_learner(
            self,
            architecture: str,
            autoencoder_type: str,
            latent_dim: int,
            number_of_blocks: int,
            filters: Union[list,int],
            kernel_sizes: Union[list,int],
            strides: Union[list,int],
            use_maxpool: Union[list,bool],
            hidden_activation: str,
            latent_activation: str,
            output_activation: str,
            global_pool_type: str,
            dropout: float,
            use_awgn: bool,
            awgn_variance: float,
            distance: str,
            similarity: str,
            similarity_c: float,
            initial_protos: np.ndarray,
            **kwargs
        ):

        ae = autoencoder_type
        
        # create the encoder
        encoder = self._create_encoder_model(
            architecture=architecture,
            latent_dim=latent_dim if ae == 'standard' else 2*latent_dim,
            number_of_blocks=number_of_blocks,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            use_maxpool=use_maxpool,
            hidden_activation=hidden_activation,
            latent_activation=latent_activation,
            global_pool_type=global_pool_type,
            dropout=dropout
        )

        # create the decoder
        decoder = self._create_decoder_model(
            encoder=encoder,
            architecture=architecture,
            latent_dim=latent_dim,
            number_of_blocks=number_of_blocks,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            use_maxpool=use_maxpool,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            dropout=dropout
        )

        # if initial prototypes are not provided, create them
        if initial_protos is None:
            initial_protos = np.random.normal(
                size=(self.number_of_properties, latent_dim)
            )

        # save prototypes as an attribute of the class
        self.prototypes = initial_protos

        # create the model
        if self.learner_type == 'domain_learner':
            model = models.DomainLearnerModel(
                encoder=encoder,
                decoder=decoder,
                initial_prototypes=initial_protos,
                latent_dimension=latent_dim,
                autoencoder_type='standard' if ae == 'standard' else \
                    'variational',
                distance=distance,
                similarity=similarity,
                similarity_c=similarity_c,
                use_awgn=use_awgn,
                awgn_variance=awgn_variance,
                name='domain_learning_model'
            )
        else:
            raise NotImplementedError("Invalid learner type.")

        # set the models as attributes of the class
        self.encoder = encoder
        self.decoder = decoder
        self.model = model

        # build the models
        self.encoder.build(
            input_shape=(None, self.height, self.width, self.channels)
        )
        self.decoder.build(
            input_shape=(None, latent_dim)
        )
        self.model.build(
            input_shape=(None, self.height, self.width, self.channels)
        )

        # set flags to indicate that the models have been created
        self.encoder_created = True
        self.decoder_created = True
        self.models_created = True
    

    def _create_encoder_model(
            self,
            architecture: str,
            latent_dim: int,
            number_of_blocks: int,
            filters: Union[list,int],
            kernel_sizes: Union[list,int],
            strides: Union[list,int],
            use_maxpool: Union[list,bool],
            hidden_activation: str,
            latent_activation: str,
            global_pool_type: str,
            dropout: float,
        ):
        if architecture == 'custom_cnn':
            encoder = models.ConvEncoder(
                input_shape=(self.height, self.width, self.channels),
                code_size=latent_dim,
                number_of_blocks=number_of_blocks,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                use_maxpool=use_maxpool,
                conv_activation=hidden_activation,
                dense_activation=latent_activation,
                global_pool_type=global_pool_type,
                dropout=dropout,
                name='encoder'
            )
        elif architecture == 'resnet18':
            encoder = models.ResNet18Encoder(
                input_shape=(self.height, self.width, self.channels),
                code_size=latent_dim,
                conv_activation=hidden_activation,
                dense_activation=latent_activation,
                global_pool_type=global_pool_type,
                dropout=dropout,
                name='encoder'
            )
        elif architecture == 'resnet34':
            encoder = models.ResNet34Encoder(
                input_shape=(self.height, self.width, self.channels),
                code_size=latent_dim,
                conv_activation=hidden_activation,
                dense_activation=latent_activation,
                global_pool_type=global_pool_type,
                dropout=dropout,
                name='encoder'
            )
        elif architecture == 'resnet50':
            encoder = models.ResNet50Encoder(
                input_shape=(self.height, self.width, self.channels),
                code_size=latent_dim,
                conv_activation=hidden_activation,
                dense_activation=latent_activation,
                global_pool_type=global_pool_type,
                dropout=dropout,
                name='encoder'
            )
        else:
            raise ValueError("Got invalid architecture.")
        
        return encoder
    
    def _create_decoder_model(
            self,
            encoder: tf.keras.Model,
            architecture: str,
            latent_dim: int,
            number_of_blocks: int,
            filters: Union[list,int],
            kernel_sizes: Union[list,int],
            strides: Union[list,int],
            use_maxpool: Union[list,bool],
            hidden_activation: str,
            output_activation: str,
            dropout: float,
            channels: Optional[int] = None
        
        ):
        if architecture == 'custom_cnn':
            decoder = models.ConvDecoder(
                encoder=encoder,
                output_channels=self.channels if channels is None else channels,
                code_size=latent_dim,
                number_of_blocks=number_of_blocks,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                use_unpooling=use_maxpool,
                deconv_activation=hidden_activation,
                output_activation=output_activation,
                dropout=dropout,
                name='decoder'
            )
        elif architecture == 'resnet18':
            decoder = models.ResNet18Decoder(
                encoder=encoder,
                output_channels=self.channels if channels is None else channels,
                code_size=latent_dim,
                deconv_activation=hidden_activation,
                output_activation=output_activation,
                dropout=dropout,
                name='decoder'
            )
        elif architecture == 'resnet34':
            decoder = models.ResNet34Decoder(
                encoder=encoder,
                output_channels=self.channels if channels is None else channels,
                code_size=latent_dim,
                deconv_activation=hidden_activation,
                output_activation=output_activation,
                dropout=dropout,
                name='decoder'
            )
        elif architecture == 'resnet50':
            decoder = models.ResNet50Decoder(
                encoder=encoder,
                output_channels=self.channels if channels is None else channels,
                code_size=latent_dim,
                deconv_activation=hidden_activation,
                output_activation=output_activation,
                dropout=dropout,
                name='decoder'
            )
        else:
            raise ValueError("Got invalid architecture.")
        
        return decoder


    def _compile_classifier_learner(
            self,
            loss: str,
            optimizer: str,
            learning_rate: float,
            weight_decay: float,
            clipnorm: float,
            clipvalue: float,
            metrics: list,
            schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
            metric_matrix: np.ndarray,
            wasserstein_lam: float,
            wasserstein_p: float,
            **kwargs
        ):

        # set the loss based on the provided string
        if loss == 'categorical_crossentropy':
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        elif loss == 'wasserstein':
            self.wass_train = True
            if metric_matrix is None:
                raise ValueError(
                    "Must provide a metric matrix for Wasserstein loss."
                )
            # make sure the output act is softmax
            if self.output_activation != 'softmax':
                raise ValueError(
                    "output_activation (set with .create_learner) must be" + \
                    " 'softmax' for Wasserstein loss."
                )
        else:
            raise ValueError(f"Got invalid loss function: {loss}.")
        
        # set the optimizer based on the provided string
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate if schedule is None else schedule,
                weight_decay=weight_decay,
                clipnorm=clipnorm,
                clipvalue=clipvalue
            )
        else:
            raise ValueError(f"Got invalid optimizer: {optimizer}.")

        # compile the model with the model's .compile method
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            metric_matrix=metric_matrix,
            wasserstein_lam=wasserstein_lam,
            wasserstein_p=wasserstein_p
        )

        # set flag to indicate that models have been compiled
        self.models_compiled = True

    def _compile_autoencoder_learner(
            self,
            loss: str,
            optimizer: str,
            learning_rate: float,
            weight_decay: float,
            clipnorm: float,
            clipvalue: float,
            metrics: list,
            lam: float,
            schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
            scaled_prior: bool,
            **kwargs
        ):
        
        # set the loss based on the provided string
        if loss == 'mse':
            loss = tf.keras.losses.MeanSquaredError()
        elif loss == 'ssim':
            def SSIMLoss(y_true, y_pred):
                y_true = (y_true + 1) / 2
                y_pred = (y_pred + 1) / 2
                return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
            loss = SSIMLoss
        else:
            raise ValueError("Got invalid loss function.")
        
        # set the optimizer based on the provided string
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate if schedule is None else schedule,
                weight_decay=weight_decay,
                clipnorm=clipnorm,
                clipvalue=clipvalue
            )
        else:
            raise ValueError("Got invalid optimizer.")

        # compile the model with the model's .compile method
        if self.autoencoder_type == 'standard':
            self.model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=metrics
            )
        elif self.autoencoder_type == 'variational':
            self.model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=metrics,
                lam=lam,
                scaled_prior=scaled_prior
            )
        else:
            raise ValueError("Got invalid autoencoder type.")

        # set flag to indicate that models have been compiled
        self.models_compiled = True
    
    def _compile_semantic_learner(
            self,
            loss: str,
            optimizer: str,
            learning_rate: float,
            weight_decay: float,
            clipnorm: float,
            clipvalue: float,
            alpha: float,
            beta: float,
            lam: float,
            metrics: list,
            schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
            metric_matrix: np.ndarray,
            wasserstein_lam: float,
            wasserstein_p: float,
            scaled_prior: bool,
            **kwargs
        ):

        # check if wasserstein loss is being used
        if loss == 'wasserstein':
            self.wass_train = True

        # set the optimizer based on the provided string
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate if schedule is None else schedule,
                weight_decay=weight_decay,
                clipnorm=clipnorm,
                clipvalue=clipvalue
            )
        else:
            raise ValueError("Got invalid optimizer.")
        
        # compile the model with the model's .compile method
        if self.learner_type == 'domain_learner':
            self.model.compile(
                loss=loss,
                optimizer=optimizer,
                alpha=alpha,
                beta=beta,
                lam=lam,
                metrics=metrics,
                metric_matrix=metric_matrix,
                wasserstein_lam=wasserstein_lam,
                wasserstein_p=wasserstein_p,
                scaled_prior=scaled_prior
            )
        else:
            raise ValueError("Invalid learner type.")

        # set flag to indicate that models have been compiled
        self.models_compiled = True
        
    
    def _train_classifier_model(
            self,
            epochs: int,
            steps_per_epoch: Optional[int],
            validation_steps: Optional[int],
            callbacks: Optional[list],
            verbose: Optional[int],
            **kwargs
        ):
        
        if not self.wass_train:
            # train the model with the model's .fit method
            history = self.model.fit(
                x=self.training_loader,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.validation_loader,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=verbose
            )
            self.training_history = history.history
        else:
            # train with a custom training loop
            trainer = cstrain.WassersteinClassifierTrainer(self.model)
            history = trainer.fit(
                training_loader=self.training_loader,
                validation_loader=self.validation_loader,
                epochs=epochs,
                batch_size=self.batch_size,
                train_size=self.train_size,
                valid_size=self.valid_size,
            )
            self.training_history = history

        # set flag to indicate that models have been trained
        self.models_trained = True
    
    def _train_autoencoder_model(
            self,
            epochs: int,
            steps_per_epoch: Optional[int],
            validation_steps: Optional[int],
            callbacks: Optional[list],
            verbose: Optional[int],
            **kwargs
        ):
        
        # train the model with the model's .fit method
        history = self.model.fit(
            x=self.training_loader,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.validation_loader,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose
        )

        # save the training history dictionary as an attribute
        self.training_history = history.history

        # set flag to indicate that models have been trained
        self.models_trained = True
    
    def _train_semantic_model(
            self,
            epochs: int,
            steps_per_epoch: Optional[int],
            validation_steps: Optional[int],
            callbacks: Optional[list],
            verbose: Optional[int],
            proto_update_type: Optional[str],
            proto_update_step_size: Optional[int],
            mu: Optional[float],
            warmup: Optional[int],
            proto_plot_save_path: Optional[str],
            proto_plot_colors: Optional[list],
            proto_plot_legend: Optional[list],
            fixed_prototypes: Optional[bool],
            **kwargs
        ):

        if self.wass_train:
            trainer = cstrain.WassersteinDomainLearnerTrainer(
                self.model,
                self.encoder,
                self.autoencoder_type
            )
            history = trainer.fit(
                training_loader=self.training_loader,
                validation_loader=self.validation_loader,
                epochs=epochs,
                batch_size=self.batch_size,
                train_size=self.train_size,
                valid_size=self.valid_size,
                warmup=warmup,
                mu=mu,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                proto_update_step_size=proto_update_step_size,
                fixed_prototypes=fixed_prototypes
            )
            self.prototypes = self.model.protos.numpy()
            self.training_history = history
        else:
            self._domain_learner_basic_train_loop(
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=verbose,
                proto_update_type=proto_update_type,
                proto_update_step_size=proto_update_step_size,
                mu=mu,
                warmup=warmup,
                proto_plot_save_path=proto_plot_save_path,
                proto_plot_colors=proto_plot_colors,
                proto_plot_legend=proto_plot_legend,
                fixed_prototypes=fixed_prototypes
            )

        # set flag to indicate that models have been trained
        self.models_trained = True


    def _setup_comet_experiment(
            self,
            log_experiment: bool
        ):
        """
        Method for setting up the comet ML experiment context (or lack thereof).
        """
        # if logging is specified, set up the experiment context
        if log_experiment:
            if comet_imported == True:
                # load in the comet information
                try:
                    with open('comet_info.json', 'r') as f:
                        comet_info = json.load(f)
                except:
                    raise ValueError("Could not load comet info from " \
                        + "'comet_info.json'.")

                experiment = Experiment(
                    api_key = comet_info['API_KEY'],
                    project_name = comet_info['PROJECT'],
                    workspace = comet_info['WORKSPACE'],
                    auto_metric_step_rate = 100,
                    auto_histogram_epoch_rate=10,
                    auto_histogram_weight_logging = False,
                    auto_histogram_gradient_logging = False,
                    auto_histogram_activation_logging = False
                )
                training_context = experiment.train()
            else:
                training_context = nullcontext()
        # otherwise, just use a null context
        else:
            training_context = nullcontext()

        return training_context

    def _domain_learner_basic_train_loop(
            self,
            epochs: int,
            steps_per_epoch: Optional[int] = None,
            validation_steps: Optional[int] = None,
            callbacks: Optional[list] = None,
            verbose: Optional[int] = 1,
            proto_update_type: Optional[str] = 'average',
            proto_update_step_size: Optional[int] = 100,
            mu: Optional[float] = 0.1,
            warmup: Optional[int] = 0,
            proto_plot_save_path: Optional[str] = None,
            proto_plot_colors: Optional[list] = None,
            proto_plot_legend: Optional[list] = None,
            fixed_prototypes: Optional[bool] = False
        ):
        """
        Internal method for training the domain learner model.
        """
        self.training_history = {}

        # initialize the prototype plotter, if specified
        if not fixed_prototypes and proto_plot_save_path is not None:
            save_path = utils.get_unused_name(proto_plot_save_path)
            os.mkdir(save_path)
            prototype_plotter = vis.PrototypePlotter2D(
                initial_prototypes=self.prototypes,
                save_directory=save_path,
                dataset=self.dataset,
                show=False,
                colors=proto_plot_colors,
                legend=proto_plot_legend
            )
        else:
            prototype_plotter = None
        
        # main training loop
        for epoch in range(epochs):
            if verbose > 0:
                if epoch < warmup:
                    print(f'\n\n\nEpoch {epoch+1}/{epochs} (warmup)')
                else:
                    print(f'\n\n\nEpoch {epoch+1}/{epochs}')

            # update the protos during the warmup stage (with mu=0.0)
            if not fixed_prototypes:
                if (warmup > 0) and (epoch <= warmup) and (epoch > 0):
                    self._domain_learner_update_prototypes(
                        mu=0.0,
                        proto_update_type=proto_update_type,
                        batches=proto_update_step_size,
                        verbose=True if verbose==1 else False
                    )
                    # save the prototype plots, if specified
                    if prototype_plotter is not None:
                        prototype_plotter.update_and_save(self.prototypes)

            # update the model
            history = self._domain_learner_update_model(
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                current_epoch=epoch,
                callbacks=callbacks,
                verbose=verbose,
                warmup=warmup
            )

            # update the training history dictionary
            for key in history.history.keys():
                if epoch == 0:
                    self.training_history[key] = [history.history[key][0]]
                else:
                    self.training_history[key].append(history.history[key][0])

            # update the prototypes (if not in warmup stage)
            if not fixed_prototypes and epoch > warmup-1:
                self._domain_learner_update_prototypes(
                    mu=mu,
                    proto_update_type=proto_update_type,
                    batches=proto_update_step_size,
                    verbose=True if verbose==1 else False
                )
                # save the prototype plots, if specified
                if prototype_plotter is not None:
                    prototype_plotter.update_and_save(self.prototypes)

    def _domain_learner_update_model(
            self,
            epochs: int = 1,
            current_epoch: int = 1,
            steps_per_epoch: Optional[int] = None,
            validation_steps: Optional[int] = None, 
            callbacks: Optional[list] = None,
            verbose: Optional[int] = 1,
            warmup: Optional[int] = 0
        ):
        """
        Internal method for updating the domain learner model.
        """
        # beta is 0 during warmup stage
        if current_epoch < warmup:
            self.model.beta.assign(0.0)
        elif current_epoch == warmup: 
            self.model.beta.assign(self.beta)

        # train the model with the model's .fit method
        history = self.model.fit(
            x=self.training_loader,
            validation_data=self.validation_loader,
            epochs=current_epoch+epochs,
            initial_epoch=current_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose
        )

        # reset the metric trackers within the model
        self.model.reset_metrics()

        # memory management
        gc.collect()

        return history

    def _domain_learner_update_prototypes(
            self,
            mu: float,
            proto_update_type: str,
            batches: int,
            verbose: bool = True
        ):
        """
        Internal method for updating the domain prototypes.
        """
        # get number of prototypes and features
        num_ps = self.number_of_properties
        num_fs = self.latent_dim

        # intialize arrays
        pred_props = np.zeros(shape=(0, num_ps))
        features_all = np.zeros(shape=(0, num_fs))

        # get number of batches if not specified (default to steps_per_epoch)
        if batches is None:
            batches = self.steps_per_epoch

        # if number of batches specified, loop through that number of batches
        i = 0
        for batch in self.training_loader.take(batches).as_numpy_iterator():
            inputs = batch[0]
            props = batch[1][1]
            features = self.encoder(inputs, training=False).numpy()
            if self.autoencoder_type == 'variational':
                features = layers.ReparameterizationLayer(
                    self.latent_dim
                )(features)[2].numpy()

            pred_props = np.append(pred_props, values=props, axis=0)
            features_all = np.append(features_all, values=features, axis=0)

            if verbose:
                print(
                    f'\rRecomputing prototypes... {i+1}/{batches}', 
                    end = ''
                )
            i += 1

        # add a line after last print statement
        if verbose:
            print()

        # convert the outputs from one-hot to a prediction vector
        pred_props_vec = np.argmax(pred_props, axis=1)

        # loop through prototypes and recompute
        for p in range(num_ps):
            # make sure that at least one p is present
            if np.any(pred_props_vec==p):
                p_feats = features_all[pred_props_vec==p,:]
                old_proto = self.prototypes[p,:]

                if proto_update_type == 'average':
                    new_proto = np.mean(p_feats, axis=0)*(1-mu) + old_proto*mu
                else:
                    raise ValueError("Invalid prototype update type.")
                
                # print feedback
                if verbose:
                    print(f'Prototype {p+1}/{num_ps} updated.')

                # update the prototype
                self.prototypes[p,:] = new_proto

            # if no p is present, just keep the old prototype
            else:
                if verbose:
                    print(f'WARNING: Prototype {p+1}/{num_ps} not updated.')
            
        # update the prototypes in the model
        self.model.protos.assign(self.prototypes)

        # add some lines for memory management
        del pred_props, features_all, batch, features, inputs, props, p_feats 
        del pred_props_vec, old_proto, new_proto
        gc.collect()