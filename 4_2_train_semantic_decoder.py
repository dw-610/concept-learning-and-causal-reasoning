"""
This script trains the semantic decoder for the SCCS-R system.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import tensorflow as tf

from modules.causal_cs.decoder import SemanticDecoder, MaskValueCallback

# ------------------------------------------------------------------------------

def main(task: str):

    DEBUG           = False
    DATA_DIR        = 'local/decoder_data/'
    MASK_VALUE      = 25.0
    EPOCHS          = 75
    LEARNING_RATE   = 1e-3
    BATCH_SIZE      = 64

    SAVE_DIR        = 'local/models/decoders/'

    if task in ['shapes', 'colors']:
        N_DIMS = 5
        N_CLASSES = 4
    elif task == 'isSpeedLimit':
        N_DIMS = 20
        N_CLASSES = 1
    else:
        raise ValueError(f'Invalid task! Got {task}')

    if DEBUG:
        tf.config.run_functions_eagerly(True)

    trn_data = np.load(DATA_DIR + f'{task}_trn_data.npy')
    trn_labels = np.load(DATA_DIR + f'{task}_trn_labels.npy')
    vld_data = np.load(DATA_DIR + f'{task}_vld_data.npy')
    vld_labels = np.load(DATA_DIR + f'{task}_vld_labels.npy')
    
    decoder = SemanticDecoder(task, N_DIMS, N_CLASSES, MASK_VALUE)
    decoder(trn_data[:10])
    decoder.summary()

    if task in ['shapes', 'colors']:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    elif task == 'isSpeedLimit':
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        raise ValueError(f'Invalid task! Got {task}')
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    decoder.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    decoder.fit(
        trn_data, trn_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(vld_data, vld_labels),
        callbacks=[MaskValueCallback()]
    )

    decoder.save(SAVE_DIR + f'{task}.keras')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    TASKS = ['shapes', 'colors', 'isSpeedLimit']

    for task in TASKS:
        print(f'\nTraining decoder for task: {task}')
        main(task)

# ------------------------------------------------------------------------------