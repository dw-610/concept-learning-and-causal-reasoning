"""
This script trains the end-to-end semantic baseline for comparing with the 
proposed SCCS-R system.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limit GPU memory to 4GB
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    except RuntimeError as e:
        print(e)

import numpy as np
import matplotlib.pyplot as plt

from modules.e2e_baseline import EndToEndSemanticModel

# ------------------------------------------------------------------------------

def main(latent_dim: int, mod_type: str):

    TASKS       = ['shapes', 'colors', 'isSpeedLimit']

    N_DECODERS  = len(TASKS)
    N_CLASSES   = [4, 4, 2]

    N_IMAGES    = 39209
    IM_SIZE     = 64
    N_CHANNELS  = 3

    DATA_DIR    = 'local/memmap_data/'
    IMAGES_PATH = DATA_DIR + 'signs_dom_trn_data.npy'

    LABELS_PATHS = [
        DATA_DIR + f'signs_dom_trn_labels_{TASK}.npy' for TASK in TASKS
    ]

    STOCH_BIN   = True
    CODE_RATE   = 1.0/8.0

    EPOCHS      = 25
    BATCH_SIZE  = 64
    EPOCHS: int             = 200
    STEPS_PER_EPOCH: int    = 200 # TRAIN_SET_SIZE // BATCH_SIZE + 1

    SCHEDULE: str           = 'cosine'
    SCH_INIT: float         = 1e-4
    SCH_WARMUP_EPOCHS: int  = 40
    SCH_WARMUP_STEPS: int   = STEPS_PER_EPOCH * SCH_WARMUP_EPOCHS
    SCH_TARGET: float       = 1e-3
    SCH_DECAY_STEPS: int    = STEPS_PER_EPOCH * (EPOCHS - SCH_WARMUP_EPOCHS)

    # --- load in the data ---

    images = np.memmap(IMAGES_PATH, dtype=np.float32, mode='r', 
                       shape=(N_IMAGES, IM_SIZE, IM_SIZE, N_CHANNELS))
    
    labels = []
    for N_CLASS, LABELS_PATH in zip(N_CLASSES, LABELS_PATHS):
        labels.append(
            np.memmap(LABELS_PATH, dtype=np.float32, mode='r', 
                      shape=(N_IMAGES, N_CLASS))
        )
    
    # --- instantiate the model ---

    model = EndToEndSemanticModel(TASKS, latent_dim, N_CLASSES, N_DECODERS, 
                                  STOCH_BIN, mod_type, CODE_RATE)
    model(images[:10])
    model.summary()

    # --- compile and train the model ---

    if SCHEDULE == 'cosine':
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=SCH_INIT,
            decay_steps=SCH_DECAY_STEPS,
            warmup_target=SCH_TARGET,
            warmup_steps=SCH_WARMUP_STEPS
        )
    elif SCHEDULE is None:
        schedule = SCH_INIT
    else:
        raise ValueError('Invalid learning rate schedule.')

    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    loss = [
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    ]

    model.compile(optimizer=optimizer, loss=loss, 
                  loss_weights=[1.0, 1.0, 1.0],
                  metrics=['accuracy', 'accuracy', 'accuracy'])

    history = model.fit(images, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # --- save the models ---

    model.encoder.save(
        f'local/models/end_to_end/len{latent_dim}_{mod_type}_encoder.keras')
    for i in range(N_DECODERS):
        if TASKS[i] in model.decoders[i].name:
            model.decoders[i].save(
                f'local/models/end_to_end/{TASKS[i]}_len{latent_dim}_{mod_type}_decoder.keras'
            )
        else:
            raise ValueError('Model name mismatch!')

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    LATENT_DIMS = [2]
    MOD_TYPES   = ['BPSK']

    for latent_dim in LATENT_DIMS:
        for mod_type in MOD_TYPES:
            main(latent_dim, mod_type)

# ------------------------------------------------------------------------------