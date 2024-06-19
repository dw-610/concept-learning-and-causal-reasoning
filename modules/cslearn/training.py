"""
This module contains classes and functions to handle the training of various
models in the CSLearn framework.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from typing import Tuple

from .arch import layers
from .arch import models

# ------------------------------------------------------------------------------

class WassersteinClassifierTrainer:
    """
    Class to handle the training of a classifier model with the custom
    Wasserstein loss.
    """
    def __init__(
            self,
            model: models.Classifier
        ):
        """
        Constructor for the WassersteinClassifierTrainer class.

        Parameters
        ----------
        model : cslearn.arch.models.Classifier
            The model to train. This should be an instance of the Classifier
            class from the cslearn.arch.models module.
            The model should have been compiled with the Wasserstein loss.
        """
        if not isinstance(model, models.Classifier):
            raise ValueError(
                "Model must be an instance of cslearn.arch.models.Classifier."
            )
        
        self.model = model
        self.optimizer = model.optimizer

        if self.optimizer is None:
            raise ValueError(
                "No optimizer detected - has the model been compiled?"
            )
        
        self.M = model.metric_matrix
        self.lam = model.wasserstein_lam
        self.p = model.wasserstein_p

        self.Kmat = tf.exp(-self.lam*self.M**self.p-1)

    @tf.function
    def train_step(
            self,
            batch_data: tf.Tensor,
            batch_labels: tf.Tensor
        ):
        """
        Method to perform a training step over a single batch of data.

        Parameters
        ----------
        batch_data : tf.Tensor
            The input data for the batch.
        batch_labels : tf.Tensor
            The true labels for the batch.

        Returns
        -------
        tf.Tensor
            The accuracy of the model on the batch.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(batch_data, training=True)
            tape.watch(predictions)
            dW_dh = get_wasserstein_gradient(
                y_true=batch_labels, 
                y_pred=predictions,
                K_matrix=self.Kmat,
                lam=self.lam
            )
        grads = tape.gradient(
            predictions,
            self.model.trainable_variables,
            output_gradients=dW_dh
        )
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(predictions, axis=-1),
                    tf.argmax(batch_labels, axis=-1)
                ),
                tf.float32
            )
        )
        return acc

    @tf.function
    def test_step(
            self,
            test_data: tf.Tensor,
            test_labels: tf.Tensor
        ):
        """
        Method to perform a testing step over some data.

        Parameters
        ----------
        test_data : tf.Tensor
            The input data for the test.
        test_labels : tf.Tensor
            The true labels for the test data.

        Returns
        -------
        tf.Tensor
            The accuracy of the model on the test data.
        """
        predictions = self.model(test_data, training=False)
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(predictions, axis=-1),
                    tf.argmax(test_labels, axis=-1)
                ),
                tf.float32
            )
        )
        return acc

    def fit(
            self,
            training_loader: tf.data.Dataset,
            validation_loader: tf.data.Dataset,
            epochs: int,
            batch_size: int,
            train_size: int,
            valid_size: int
        ) -> dict:
        """
        Method to perform the full training loop.

        Parameters
        ----------
        training_loader : tf.data.Dataset
            The training data loader.
            Data samples should be images, and labels should be one-hot encoded.
        validation_loader : tf.data.Dataset
            The validation data loader.
            Data samples should be images, and labels should be one-hot encoded.
        epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size to use.
        train_size : int
            The number of samples in the training set.
        valid_size : int
            The number of samples in the validation set.

        Returns
        -------
        dict
            A dictionary containing the training history.
            The dictionary contains the following keys:
            - 'accuracy': a list of the training accuracy at each epoch
            - 'val_accuracy': a list of the validation accuracy at each epoch
        """
        history = {'accuracy': [], 'val_accuracy': []}
        steps_per_epoch = train_size // batch_size + 1
        print('\nStarting classifier training loop with Wasserstein loss...')
        for epoch in range(epochs):

            print(f'\nEpoch: {epoch+1}/{epochs}')
            print(f'Learning rate: {self.optimizer.lr.numpy()}')
            iterator = tqdm(
                training_loader.take(steps_per_epoch),
                desc='Batches',
                ncols=80,
                total=train_size//batch_size+1
            )
            acc = 0
            for data, labels in iterator:
                acc += self.train_step(data, labels)
            acc = acc.numpy()/steps_per_epoch
            history['accuracy'].append(acc)
            print(f'Training accuracy = {np.round(acc*100,2)}%')

            print('Testing on the validation set...', end=' ')
            valid_steps = valid_size // batch_size + 1
            acc = 0
            for data, labels in validation_loader.take(valid_steps):
                acc += self.test_step(data, labels)
            acc = acc.numpy()/valid_steps
            history['val_accuracy'].append(acc)
            print(f'accuracy = {np.round(acc*100,2)}%')

        return history

# ------------------------------------------------------------------------------

class WassersteinDomainLearnerTrainer:
    """
    Class to handle the training of a domain learner model with the custom
    Wasserstein loss.
    """
    def __init__(
            self,
            model: models.DomainLearnerModel,
            encoder: tf.keras.models.Model,
            autoencoder_type: str
        ):
        """
        Constructor for the WassersteinDomainLearnerTrainer class.

        Parameters
        ----------
        model : cslearn.arch.models.DomainLearnerModel
            The model to train. This should be an instance of the
            DomainLearnerModel class from the cslearn.arch.models module.
            The model should have been compiled with the Wasserstein loss.
        encoder : tf.keras.models.Model
            The encoder model used to extract features from the input data.
        autoencoder_type : str
            The type of autoencoder used to extract features from the input data.
            Options are 'standard' and 'variational'.
        """
        if not isinstance(model, models.DomainLearnerModel):
            raise ValueError(
                "Model must be a cslearn.arch.models.DomainLearnerModel."
            )
        
        self.model = model
        self.encoder = encoder
        self.optimizer = model.optimizer
        self.ae_type = autoencoder_type
        self.scaled = model.scaled_prior

        if self.optimizer is None:
            raise ValueError(
                "No optimizer detected - has the model been compiled?"
            )
        
        self.lam = model.wasserstein_lam
        self.p = model.wasserstein_p
        self.M = model.metric_matrix**self.p

        self.Kmat = tf.exp(-self.lam*self.M-1)

    @tf.function
    def train_step(
        self,
        batch_data: tf.Tensor,
        batch_labels: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Method to perform a training step over a single batch of data.

        Parameters
        ----------
        batch_data : tf.Tensor
            The input data for the batch.
        batch_labels : list of tf.Tensor
            The true labels for the batch.
            The first element is the true images, and the second is the true
            properties.

        Returns
        -------
        tf.Tensor
            The accuracy of the model on the batch.
        tf.Tensor
            The reconstruction loss for the batch.
        """
        true_images = batch_labels[0]
        true_properties = batch_labels[1]
        with tf.GradientTape() as tape:
            if self.ae_type == 'standard':
                pred_images, pred_properties, _ = self.model(
                    batch_data, 
                    training=True
                )
                tape.watch(pred_images)
                tape.watch(pred_properties)
                dlw_dc = get_wasserstein_gradient(
                    y_true=true_properties, 
                    y_pred=pred_properties,
                    K_matrix=self.Kmat
                )
                dlr_dx = get_mse_gradient(
                    y_true=true_images,
                    y_pred=pred_images
                )
            elif self.ae_type == 'variational':
                pred_images, pred_properties, _, logstds, mus = self.model(
                    batch_data,
                    training=True
                )
                tape.watch(pred_images)
                tape.watch(pred_properties)
                tape.watch(logstds)
                tape.watch(mus)
                dlw_dc = get_wasserstein_gradient(
                    y_true=true_properties, 
                    y_pred=pred_properties,
                    K_matrix=self.Kmat
                )
                dlr_dx = get_vae_lr_gradient(
                    y_true=true_images,
                    y_pred=pred_images
                )
                dkl_dm = get_vae_kl_mu_gradient(mus, self.scaled)
                dkl_dlogstd = get_vae_kl_logstd_gradient(logstds, self.scaled)
            else:
                raise ValueError("Invalid autoencoder type.")
        if self.ae_type == 'standard':
            grads = tape.gradient(
                [pred_images, pred_properties],
                self.model.trainable_variables,
                output_gradients=[
                    tf.multiply(self.model.alpha, dlr_dx), 
                    tf.multiply(self.model.beta, dlw_dc)
                ]
            )
        elif self.ae_type == 'variational':
            grads = tape.gradient(
                [pred_images, pred_properties, mus, logstds],
                self.model.trainable_variables,
                output_gradients=[
                    tf.multiply(self.model.alpha, dlr_dx), 
                    tf.multiply(self.model.beta, dlw_dc),
                    tf.multiply(self.model.lam, dkl_dm),
                    tf.multiply(self.model.lam, dkl_dlogstd)
                ]
            )
        else:
            raise ValueError("Invalid autoencoder type.")
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(pred_properties, axis=-1),
                    tf.argmax(true_properties, axis=-1)
                ),
                tf.float32
            )
        )
        recon_loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(pred_images - true_images), axis=[1,2,3])
        )
        if self.ae_type == 'variational':
            if self.scaled:
                prior_var = tf.constant(1.0/mus.shape[1], dtype=tf.float32)
            else:
                prior_var = tf.constant(1.0, dtype=tf.float32)
            kl_loss = tf.reduce_mean(
                tf.reduce_mean(
                    tf.exp(2*logstds)/prior_var + (mus**2)/prior_var - \
                          (2*logstds-tf.math.log(prior_var)),
                    axis=-1
                )
            )
        else:
            kl_loss = None
        return acc, recon_loss, kl_loss
        
    @tf.function
    def test_step(
            self,
            test_data: tf.Tensor,
            test_labels: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Method to perform a testing step over some data.

        Parameters
        ----------
        test_data : tf.Tensor
            The input data for the test.
        test_labels : list of tf.Tensor
            The true labels for the test data.
            The first element is the true images, and the second is the true
            properties.
        
        Returns
        -------
        tf.Tensor
            The accuracy of the model on the test data.
        tf.Tensor
            The reconstruction loss for the test data.
        """
        true_images = test_labels[0]
        true_properties = test_labels[1]
        if self.ae_type == 'standard':
            pred_images, pred_properties, _ = self.model(
                test_data, 
                training=False
            )
        elif self.ae_type == 'variational':
            pred_images, pred_properties, _, logstds, mus = self.model(
                test_data,
                training=False
            )
        else:
            raise ValueError("Invalid autoencoder type.")
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(pred_properties, axis=-1),
                    tf.argmax(true_properties, axis=-1)
                ),
                tf.float32
            )
        )
        recon_loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(pred_images - true_images), axis=[1,2,3])
        )
        if self.ae_type == 'variational':
            if self.scaled:
                prior_var = tf.constant(1.0/mus.shape[1], dtype=tf.float32)
            else:
                prior_var = tf.constant(1.0, dtype=tf.float32)
            kl_loss = tf.reduce_mean(
                tf.reduce_mean(
                    tf.exp(2*logstds)/prior_var + (mus**2)/prior_var - \
                          (2*logstds-tf.math.log(prior_var)),
                    axis=-1
                )
            )
        else:
            kl_loss = None
        return acc, recon_loss, kl_loss
 
    def fit(
            self,
            training_loader: tf.data.Dataset,
            validation_loader: tf.data.Dataset,
            epochs: int,
            batch_size: int,
            train_size: int,
            valid_size: int,
            warmup: int,
            mu: float,
            steps_per_epoch: int,
            validation_steps: int,
            proto_update_step_size: int,
            fixed_prototypes: bool = False
        ) -> dict:
        """
        Method to perform the full training loop.

        Parameters
        ----------
        training_loader : tf.data.Dataset
            The training data loader.
            Data samples should be images.
            Labels is a list of multiple tensors:
            - the first tensor is the true images
            - the second tensor is the true properties
            - the third tensor is the distances to the prototypes (unused here)
        validation_loader : tf.data.Dataset
            The validation data loader.
            Data structure is the same as for the training loader.
        epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size to use.
        train_size : int
            The number of samples in the training set.
        valid_size : int
            The number of samples in the validation set.
        warmup : int
            The number of epochs to train with only reconstruction loss. 
            After this, the Wasserstein loss is added.
        mu : float
            The "mixing" parameter for the prototype update.
        steps_per_epoch : int
            The number of steps per epoch in the training data loader.
        proto_update_step_size : int
            The number of batches to use for the prototype update.
        fixed_prototypes : bool, optional
            Whether to keep the prototypes fixed during training.
            Default is False.

        Returns
        -------
        dict
            A dictionary containing the training history.
            The dictionary contains the following keys:
            - 'accuracy': a list of the training accuracy at each epoch
            - 'lr': a list of the training reconstruction loss at each epoch
            - 'val_accuracy': a list of the validation accuracy at each epoch
            - 'val_lr': a list of the validation reconstruction loss each epoch
        """
        history = {'accuracy': [], 'lr': [], 'val_accuracy': [], 'val_lr': [],
                   'kl_div': [], 'val_kl_div': []}
        if steps_per_epoch is None:
            steps_per_epoch = train_size // batch_size + 1
        dataset_iter = iter(training_loader)
        valid_iter = iter(validation_loader.repeat())
        print('\nStarting training for domain learner with Wasserstein loss...')
        for epoch in range(epochs):
            print(f'\nEpoch: {epoch+1}/{epochs}')
            print(f'Learning rate: {self.optimizer.learning_rate.numpy()}')

            # set the beta parameter based on the current epoch
            if epoch < warmup:
                self.model.beta.assign(0.0)
            elif epoch == warmup:
                self.model.beta.assign(self.model.beta_val)
            
            # training
            acc = 0
            recon_loss = 0
            kl_div = 0
            for step in tqdm(range(steps_per_epoch), desc='Batches', ncols=80):
                data, labels = next(dataset_iter)
                a, l, kl = self.train_step(data, labels)
                acc += a
                recon_loss += l
                kl_div += kl if kl is not None else 0
            acc = acc.numpy()/steps_per_epoch
            recon_loss = recon_loss.numpy()/steps_per_epoch
            kl_div = kl_div.numpy()/steps_per_epoch if kl_div != 0 else None
            history['accuracy'].append(acc)
            history['lr'].append(recon_loss)
            if kl is not None:
                history['kl_div'].append(kl_div)
            print(f'Training accuracy = {np.round(acc*100,2)}%')
            print(f'Training reconstruction loss = {np.round(recon_loss,2)}')
            if kl is not None:
                print(f'Training KL divergence = {np.round(kl_div,2)}')

            # validation
            print('Testing on the validation set...', end=' ')
            if validation_steps is None:
                validation_steps = valid_size // batch_size + 1
            acc = 0
            recon_loss = 0
            kl_div = 0
            for step in range(validation_steps):
                data, labels = next(valid_iter)
                a, l, kl = self.test_step(data, labels)
                acc += a
                recon_loss += l
                kl_div += kl if kl is not None else 0
            acc = acc.numpy()/validation_steps
            recon_loss = recon_loss.numpy()/validation_steps
            kl_div = kl_div.numpy()/validation_steps if kl_div != 0 else None
            history['val_accuracy'].append(acc)
            history['val_lr'].append(recon_loss)
            if kl is not None:
                history['val_kl_div'].append(kl_div)
            print(f'accuracy = {np.round(acc*100,2)}%')
            print(f'reconstruction loss = {np.round(recon_loss,2)}')
            if kl is not None:
                print(f'KL divergence = {np.round(kl_div,2)}')

            # update prototypes
            if not fixed_prototypes:
                new_protos = domain_learner_update_prototypes(
                    old_prototypes=self.model.protos.numpy(),
                    encoder=self.encoder,
                    training_loader=training_loader,
                    autoencoder_type=self.ae_type,
                    mu=0.0 if epoch < warmup else mu,
                    proto_update_type='average',
                    batches=proto_update_step_size,
                    steps_per_epoch=steps_per_epoch,
                    verbose=True
                )
                self.model.protos.assign(new_protos)
            
            # if no M given, update M with new prototypes
            # - this uses the Euclidean distance for now
            # - can generalize to work with other metrics later
            if not self.model.M_fixed:
                np_protos = self.model.protos.numpy()
                diff = np_protos[:,np.newaxis,:] - np_protos[np.newaxis,:,:]
                dist = np.sqrt(np.sum(diff**2, axis=-1))
                self.M = dist**self.p
                self.Kmat = tf.exp(-self.lam*self.M-1)

        return history

# ------------------------------------------------------------------------------

def get_wasserstein_gradient(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        K_matrix: tf.Tensor,
        lam: float = 1.0,
    ) -> tf.Tensor:
    """
    Function computing the gradient of the Wasserstein loss for the given true
    labels and predictions.

    The method used here is based on "Learning with a Wasserstein Loss" by
    Frogner et al. (2015), specifically Algorithm 1. Specifically, Sinkhorn
    iterations are used to compute the gradient.

    Parameters
    ----------
    y_true : tf.Tensor
        True labels, represented as a (batch of) histogram vector(s), e.g., the
        output of the softmax function.
    y_pred : tf.Tensor
        A (batch of) predicted label(s), represented as a histogram vector(s).
    K_matrix : tf.Tensor
        Matrix exponential of -lam*M-1, where M is the ground metric matrix.
    lam : float, optional
        Balancing parameter between the true total cost and entropic
        regularization.
        Default is 1.0.

    Returns
    -------
    tf.Tensor
        The gradient of the Wasserstein loss.
        This is a tensor of the same shape as y_true and y_pred, i.e., it is the
        gradient of the loss for *each sample* in the batch (not aggregated).
    """
    dim = y_true.shape[1]
    u = tf.ones_like(y_true)

    diff = tf.constant(float('inf'), dtype=tf.float32)
    while diff > 1e-3:
        u_ = u
        x = tf.matmul(u, K_matrix)
        x = tf.divide(y_true, x + 1e-16)
        x = tf.matmul(x, tf.transpose(K_matrix))
        u = tf.divide(y_pred, x + 1e-16) + 1e-16
        diff = tf.reduce_max(tf.reduce_sum(tf.square(u - u_),axis=-1))

    term1 = tf.math.log(u)/lam
    term2 = tf.repeat(
        tf.math.log(tf.reduce_sum(u, axis=1, keepdims=True))/dim/lam,
        dim,
        axis=1
    )

    grad = term1 + term2
    return grad

# ------------------------------------------------------------------------------

def get_mse_gradient(
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
    """
    This function computes the gradient of the image reconstruction loss term
    for the given true images and predictions, given that the loss is the mean
    squared error.

    The resulting gradient is that of the loss with respect to the output image
    variables, which can the be used to backpropagate through the model.

    Parameters
    ----------
    y_true : tf.Tensor
        True images.
    y_pred : tf.Tensor
        Predicted images.

    Returns
    -------
    tf.Tensor
        The gradient of the image reconstruction loss. This is a tensor of the
        same shape as y_true and y_pred, i.e., it is the gradient of the loss
        for *each sample* in the batch (not aggregated).
    """
    return 2*(y_pred - y_true)

# ------------------------------------------------------------------------------

def get_vae_lr_gradient(
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
    """
    This function computes the gradient of the reconstruction loss term of the 
    overall ELBO loss for the VAE.

    The resulting gradient is that of the loss with respect to the output image
    variables, which can the be used to backpropagate through the model.

    Parameters
    ----------
    y_true : tf.Tensor
        True images.
    y_pred : tf.Tensor
        Predicted images.

    Returns
    -------
    tf.Tensor
        The gradient of the image reconstruction loss. This is a tensor of the
        same shape as y_true and y_pred, i.e., it is the gradient of the loss
        for *each sample* in the batch (not aggregated).
    """
    return 2*(y_pred - y_true)

# ------------------------------------------------------------------------------

def get_vae_kl_mu_gradient(
        mu: tf.Tensor,
        scaled_prior: bool
    ) -> tf.Tensor:
    """
    This function computes the gradient of the KL divergence term of the overall
    ELBO loss for the VAE with respect to the mean parameter.

    The resulting gradient is that of the loss with respect to the mean
    parameter, which can then be used to backpropagate through the model.

    Parameters
    ----------
    mu : tf.Tensor
        The mean parameter of the VAE.
    scaled_prior : bool
        Whether to use the scaled prior for the KL divergence term.

    Returns
    -------
    tf.Tensor
        The gradient of the KL divergence loss with respect to the mean
        parameter. This is a tensor of the same shape as mu, i.e., it is the
        gradient of the loss for *each sample* in the batch (not aggregated).
    """
    if scaled_prior:
        prior_var = tf.constant(1.0/mu.shape[1], dtype=tf.float32)
    else:
        prior_var = tf.constant(1.0, dtype=tf.float32)
    return 2*mu/prior_var

# ------------------------------------------------------------------------------

def get_vae_kl_logstd_gradient(
        logstd: tf.Tensor,
        scaled_prior: bool
    ) -> tf.Tensor:
    """
    This function computes the gradient of the KL divergence term of the overall
    ELBO loss for the VAE with respect to the log standard deviation parameter.

    The resulting gradient is that of the loss with respect to the log standard
    deviation parameter, which can then be used to backpropagate through the
    model.

    Parameters
    ----------
    logstd : tf.Tensor
        The log standard deviation parameter of the VAE.
    scaled_prior : bool
        Whether to use the scaled prior for the KL divergence term.

    Returns
    -------
    tf.Tensor
        The gradient of the KL divergence loss with respect to the log standard
        deviation parameter. This is a tensor of the same shape as logstd, i.e.,
        it is the gradient of the loss for *each sample* in the batch (not
        aggregated).
    """
    if scaled_prior:
        prior_var = tf.constant(1.0/logstd.shape[1], dtype=tf.float32)
    else:
        prior_var = tf.constant(1.0, dtype=tf.float32)
    return 2*(tf.exp(2*logstd)/prior_var - 1)

# ------------------------------------------------------------------------------

def domain_learner_update_prototypes(
        old_prototypes: np.ndarray,
        encoder: tf.keras.models.Model,
        training_loader: tf.data.Dataset,
        autoencoder_type: str,
        mu: float,
        proto_update_type: str,
        batches: int,
        steps_per_epoch: int,
        verbose: bool = True,
    ):
    """
    Function to update the prototypes of a domain learner model based on the
    current state of the model.

    Parameters
    ----------
    old_prototypes : np.ndarray
        The current prototypes of the domain learner model.
    encoder : tf.keras.models.Model
        The encoder model used to extract features from the input data.
    training_loader : tf.data.Dataset
        The training data loader.
        Data samples should be images.
        Labels is a list of multiple tensors:
        - the first tensor is the true images (unused here)
        - the second tensor is the true properties
        - the third tensor is the distances to the prototypes (unused here)
    autoencoder_type : str
        The type of autoencoder used to extract features from the input data.
        Currently, only 'standard' is supported.
    mu : float
        The "mixing" parameter for the prototype update.
    proto_update_type : str
        The type of prototype update to perform.
        Currently, only 'average' is supported.
    batches : int
        The number of batches to use for the prototype update.
    steps_per_epoch : int
        The number of steps per epoch in the training data loader.
    verbose : bool, optional
        Whether to print feedback during the prototype update.
        Default is True.
    """
    # get number of prototypes and features
    num_ps = old_prototypes.shape[0]
    num_fs = old_prototypes.shape[1]

    # intialize arrays
    pred_props = np.zeros(shape=(0, num_ps))
    features_all = np.zeros(shape=(0, num_fs))

    # get number of batches if not specified (default to steps_per_epoch)
    if batches is None:
        batches = steps_per_epoch

    # if number of batches specified, loop through that number of batches
    i = 0
    for batch in training_loader.take(batches).as_numpy_iterator():
        inputs = batch[0]
        props = batch[1][1]
        features = encoder(inputs, training=False).numpy()
        if autoencoder_type == 'variational':
            features = layers.ReparameterizationLayer(
                num_fs
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
    new_prototypes = np.copy(old_prototypes)
    for p in range(num_ps):
        # make sure that at least one p is present
        if np.any(pred_props_vec==p):
            p_feats = features_all[pred_props_vec==p,:]
            old_proto = old_prototypes[p,:]

            if proto_update_type == 'average':
                new_proto = np.mean(p_feats, axis=0)*(1-mu) + old_proto*mu
            else:
                raise ValueError("Invalid prototype update type.")
            
            # print feedback
            if verbose:
                print(f'Prototype {p+1}/{num_ps} updated.')

            # update the prototype
            new_prototypes[p,:] = new_proto

        # if no p is present, just keep the old prototype
        else:
            if verbose:
                print(f'WARNING: Prototype {p+1}/{num_ps} not updated.')

    return new_prototypes

# ------------------------------------------------------------------------------