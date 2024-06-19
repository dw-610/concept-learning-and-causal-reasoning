"""
This script trains a domain learner on the sign data.
"""

# ------------------------------------------------------------------------------
# imports

import os
import numpy as np

from modules.cslearn.controllers import ImageLearningController

# ------------------------------------------------------------------------------

def main(domain: str):

    TRAIN_SET_SIZE: int     = 39209
    VALID_SET_SIZE: int     = 2630

    IM_SIZE: int            = 64
    NUM_CHANNELS: int       = 3
    NUM_CLASSES: int        = 4

    DATA_DIR: str          = 'local/memmap_data/'

    PATHS_DICT: dict = {
        'train_data_path': DATA_DIR + 'signs_dom_trn_data.npy',
        'train_labels_path': DATA_DIR + f'signs_dom_trn_labels_{domain}.npy',
        'valid_data_path': DATA_DIR + 'signs_tst_data.npy',
        'valid_labels_path': DATA_DIR + f'signs_tst_labels_{domain}.npy'
    }

    SHAPES_DICT: dict = {
        'train_data_shape': (TRAIN_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'train_labels_shape': (TRAIN_SET_SIZE, NUM_CLASSES),
        'valid_data_shape': (VALID_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'valid_labels_shape': (VALID_SET_SIZE, NUM_CLASSES)
    }

    # --- hyperparameters ---

    BATCH_SIZE: int         = 64

    LATENT_DIM: int         = 1
    ARCH: str               = 'custom_cnn'
    AE_TYPE: str            = 'variational'
    GLOBAL_POOL_TYPE: str   = 'max'
    NUM_BLOCKS: int         = 5
    FILTERS: list           = [16, 32, 64, 128, 256]
    KERNEL_SIZES: list      = [3, 3, 3, 3, 3]
    STRIDES: list           = [2, 2, 2, 2, 2]
    DROPOUT: float          = 0.0

    EPOCHS: int             = 75
    STEPS_PER_EPOCH: int    = 200
    PROTO_STEP_SIZE: int    = 200
    MU: float               = 0.75
    WARMUP: int             = 1

    LOSS: str               = 'wasserstein'
    SCHEDULE: str           = 'cosine'
    SCH_INIT: float         = 1e-4
    SCH_WARMUP_EPOCHS: int  = 15
    SCH_WARMUP_STEPS: int   = STEPS_PER_EPOCH * SCH_WARMUP_EPOCHS
    SCH_TARGET: float       = 1e-3
    SCH_DECAY_STEPS: int    = STEPS_PER_EPOCH * (EPOCHS - SCH_WARMUP_EPOCHS)
    ALPHA: float            = 0.5/IM_SIZE/IM_SIZE/NUM_CHANNELS
    BETA: float             = 2.0
    LAM: float              = 0.1/LATENT_DIM

    if domain == 'shapes':
        METRIC_M: np.ndarray    = np.array(
            [
                [0.0, 2.0, 4.0, 5.0],
                [2.0, 0.0, 2.0, 3.0],
                [4.0, 2.0, 0.0, 1.0],
                [5.0, 3.0, 1.0, 0.0]
            ]
        ).astype(np.float32)/2.5
    elif domain == 'colors':
        METRIC_M: np.ndarray    = np.array(
            [
                [0.0, 2.0, 1.0, 1.5],
                [2.0, 0.0, 3.0, 1.5],
                [1.0, 3.0, 0.0, 1.5],
                [1.5, 1.5, 1.5, 0.0]
            ]
        ).astype(np.float32)/2.0
    else:
        raise ValueError('Invalid domain.')
    WASS_P: float           = 1.0
    WASS_LAM: float         = 2.0

    # --- create the models/ directory if it doesn't exist ---

    if not os.path.exists('local/models/'):
        os.makedirs('local/models/')

    # --- use CSLearn to train the domain learner ---
    
    ctrl = ImageLearningController(learner_type='domain_learner')

    ctrl.create_data_loaders(dataset='local', batch_size=BATCH_SIZE,
                             paths_dict=PATHS_DICT, shapes_dict=SHAPES_DICT)

    ctrl.create_learner(latent_dim=LATENT_DIM, architecture=ARCH, 
                        autoencoder_type=AE_TYPE, 
                        global_pool_type=GLOBAL_POOL_TYPE, 
                        number_of_blocks=NUM_BLOCKS, filters=FILTERS, 
                        kernel_sizes=KERNEL_SIZES, strides=STRIDES,
                        dropout=DROPOUT)

    ctrl.compile_learner(loss=LOSS, schedule_type=SCHEDULE, sch_init_lr=SCH_INIT,
                         sch_warmup_steps=SCH_WARMUP_STEPS,
                         sch_warmup_target=SCH_TARGET,
                         sch_decay_steps=SCH_DECAY_STEPS, alpha=ALPHA, beta=BETA,
                         lam=LAM, metric_matrix=METRIC_M,
                         wasserstein_lam=WASS_LAM, wasserstein_p=WASS_P)

    ctrl.encoder.summary()
    ctrl.decoder.summary()
    ctrl.model.summary()

    ctrl.train_learner(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                       proto_update_step_size=PROTO_STEP_SIZE, mu=MU, 
                       warmup=WARMUP)

    ctrl.save_models('local/models/')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    print('\n\nTraining domain learner on shapes data.')
    main('shapes')

    print('\n\nTraining domain learner on colors data.')
    main('colors')

# ------------------------------------------------------------------------------