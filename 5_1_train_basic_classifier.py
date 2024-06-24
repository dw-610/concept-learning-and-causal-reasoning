"""
This script trains a basic image classifier on the sign data.

This classifier will be used in the technical and effective baseline systems.
"""

# ------------------------------------------------------------------------------
# imports

import os
import keras.backend as K

from modules.cslearn.controllers import ImageLearningController

# ------------------------------------------------------------------------------
# main

def main(task: str):

    TRAIN_SET_SIZE: int     = 39209
    VALID_SET_SIZE: int     = 2630

    IM_SIZE: int            = 64
    NUM_CHANNELS: int       = 3

    BATCH_SIZE: int         = 64
    EPOCHS: int             = 200
    STEPS_PER_EPOCH: int    = 200 # TRAIN_SET_SIZE // BATCH_SIZE + 1

    LATENT_DIM: int         = 20
    ARCH: str               = 'custom_cnn'
    GLOBAL_POOL_TYPE: str   = 'max'
    NUM_BLOCKS: int         = 5
    FILTERS: list           = [48, 96, 192, 384, 768]
    KERNEL_SIZES: list      = [3, 3, 3, 3, 3]
    STRIDES: list           = [2, 2, 2, 2, 2]
    DROPOUT: float          = 0.0

    LOSS: str               = 'categorical_crossentropy'
    SCHEDULE: str           = 'cosine'
    SCH_INIT: float         = 1e-4
    SCH_WARMUP_EPOCHS: int  = 40
    SCH_WARMUP_STEPS: int   = STEPS_PER_EPOCH * SCH_WARMUP_EPOCHS
    SCH_TARGET: float       = 1e-3
    SCH_DECAY_STEPS: int    = STEPS_PER_EPOCH * (EPOCHS - SCH_WARMUP_EPOCHS)

    METRICS: list           = ['accuracy']

    if task in ['shapes', 'colors']:
        NUM_CLASSES = 4
    elif task == 'isSpeedLimit':
        NUM_CLASSES = 2
    else:
        raise ValueError('Invalid task.')
    
    paths_dict: dict = {
        'train_data_path': 'local/memmap_data/signs_dom_trn_data.npy',
        'train_labels_path': f'local/memmap_data/signs_dom_trn_labels_{task}.npy',
        'valid_data_path': 'local/memmap_data/signs_tst_data.npy',
        'valid_labels_path': f'local/memmap_data/signs_tst_labels_{task}.npy'
    }

    shapes_dict: dict = {
        'train_data_shape': (TRAIN_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'train_labels_shape': (TRAIN_SET_SIZE, NUM_CLASSES),
        'valid_data_shape': (VALID_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'valid_labels_shape': (VALID_SET_SIZE, NUM_CLASSES)
    }

    if not os.path.exists('local/models/classifiers'):
        os.makedirs('local/models/classifiers')
    
    ctrl = ImageLearningController(learner_type='classifier')

    ctrl.create_data_loaders(
        dataset='local',
        batch_size=BATCH_SIZE,
        paths_dict=paths_dict,
        shapes_dict=shapes_dict,
        buffer_size=1000
    )

    ctrl.create_learner(
        latent_dim=LATENT_DIM,
        architecture=ARCH,
        global_pool_type=GLOBAL_POOL_TYPE,
        number_of_blocks=NUM_BLOCKS,
        filters=FILTERS,
        kernel_sizes=KERNEL_SIZES,
        strides=STRIDES,
        dropout=DROPOUT
    )

    ctrl.compile_learner(
        loss=LOSS,
        schedule_type=SCHEDULE,
        sch_init_lr=SCH_INIT,
        sch_warmup_steps=SCH_WARMUP_STEPS,
        sch_warmup_target=SCH_TARGET,
        sch_decay_steps=SCH_DECAY_STEPS,
        metrics=METRICS
    )

    ctrl.encoder.summary()
    ctrl.model.summary()

    ctrl.train_learner(
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH
    )

    ctrl.model.save(f'local/models/classifiers/{task}.keras')

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    TASKS = ['isSpeedLimit']

    for task in TASKS:
        main(task)
        K.clear_session()

# ------------------------------------------------------------------------------