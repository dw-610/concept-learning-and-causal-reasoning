"""
This module contains code relating to the semantic decoder for the SCCS-R system.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf

# ------------------------------------------------------------------------------

class MaskValueCallback(tf.keras.callbacks.Callback):
    """A callback to log the mask value of the decoder."""
    def on_epoch_begin(self, epoch, logs=None):
        mask_value = self.model.layers[0].mask_value.numpy()
        print(f'Epoch {epoch + 1}: Mask value is {mask_value}')

# ------------------------------------------------------------------------------

class MaskingLayer(tf.keras.layers.Layer):
    """
    A custom masking layer that will replace NaN values in the input data with
    a trainable parameter to handle the missing data.
    """
    def __init__(self, mask_value=0.0, **kwargs):
        """Constructor for the MaskingLayer class."""
        super(MaskingLayer, self).__init__(**kwargs)
        self.mask_value = self.add_weight(
            name='mask_value',
            shape=(),
            initializer=tf.constant_initializer(mask_value),
            trainable=True
        )

    def call(self, inputs):
        """Call method for the MaskingLayer class."""
        outputs = tf.where(
            tf.math.is_nan(inputs), 
            self.mask_value*tf.ones_like(inputs), 
            inputs
        )
        return outputs
    
    def get_output_shape_for(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(MaskingLayer, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config
    
# ------------------------------------------------------------------------------

class SemanticDecoder(tf.keras.models.Model):
    """
    The semantic decoder model for the SCCS-R system.

    This model will take in the semantic data that
    was received and decode it to perform the task. The input data can contain
    NaN values (due to reasoner dropping values). This model will replace these
    values with a trainable parameter to handle the missing data.
    """
    def __init__(self, task: str, n_dims: int, n_classes: int, mask_value=0.0, 
                 **kwargs):
        """Constructor for the SemanticDecoder class."""
        super(SemanticDecoder, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.n_dims = n_dims
        self.n_classes = n_classes
        self.task = task

        self.name = task + '_decoder'

        self.layers_list = []

        self.layers_list.append(MaskingLayer(self.mask_value))
        self.layers_list.append(tf.keras.layers.Dense(32, activation='relu'))
        self.layers_list.append(tf.keras.layers.Dense(32, activation='relu'))
        self.layers_list.append(tf.keras.layers.Dense(32, activation='relu'))
        self.layers_list.append(tf.keras.layers.Dense(32, activation='relu'))
        if n_classes == 1:
            self.layers_list.append(
                tf.keras.layers.Dense(n_classes, activation='sigmoid')
            )
        else:
            self.layers_list.append(
                tf.keras.layers.Dense(n_classes, activation='softmax')
            )

    def call(self, inputs):
        """Call method for the SemanticDecoder class."""
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x
    
    def summary(self):
        x = tf.keras.Input(shape=(self.n_dims,))
        model = tf.keras.models.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
    def on_epoch_end(self, epoch):
        print(f'Epoch {epoch} ended')
    
    def get_config(self):
        config = super(SemanticDecoder, self).get_config()
        config.update({
            'name': self.name,
            'task': self.task,
            'mask_value': self.mask_value,
            'n_dims': self.n_dims,
            'n_classes': self.n_classes
        })
        return config
    
# ------------------------------------------------------------------------------