"""
This module defines some custom callbacks for use with model.fit().
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf

Callback = tf.keras.callbacks.Callback

# ------------------------------------------------------------------------------

class LearningRateLogger(Callback):
    """
    This callback prints the learning rate at the beginning of each epoch.

    It has no parameters.
    """
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)
        print(f"\nEpoch {epoch + 1}: Learning rate is {lr.numpy()}")

# ------------------------------------------------------------------------------
        
class ImageLoggerCallback(Callback):
    """
    This callback logs images produced by the decoded to comet ML.

    Note that the learner must have a decoder model, i.e. it cannot be a 
    classifier.

    Parameters
    ----------
    experiment : comet_ml.Experiment
        The experiment object to log the images to.
    validation_feeder : tf.data.Dataset
        The validation dataset to get images from.
    autoencoder : tf.keras.Model
        The autoencoder model to use for producing the images.
    """
    def __init__(self, experiment, validation_feeder, autoencoder):
        super(ImageLoggerCallback, self).__init__()
        self.vld_feeder = validation_feeder
        self.autoencoder = autoencoder
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        batches = self.vld_feeder.batch(1)
        i = 0
        for batch in batches:
            images = batch[0].numpy()
            break
        images = images[0,:,:,:,:]
        preds= self.autoencoder(images, training=False).numpy()
        for i, image in enumerate(images):
            self.experiment.log_image(image, name=f'{i}_image')
            self.experiment.log_image(preds[i], name=f'{i}_predicted')
            if i == 9:
                break

# ------------------------------------------------------------------------------