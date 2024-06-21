"""
This script uses the trained semantic encoders to build a dataset that will be
used for the following causal learning tasks.
"""

# ------------------------------------------------------------------------------
# imports

import os
import numpy as np
import pandas as pd
from keras import backend as K

from modules.cslearn.controllers import ImageLearningController
from modules.cslearn.arch import layers as layers

# ------------------------------------------------------------------------------

def main(set_type: str):

    df = pd.DataFrame()

    TRAIN_SET_SIZE: int     = 39209
    if set_type == 'train':
        VALID_SET_SIZE: int     = 10000
    elif set_type == 'test':
        VALID_SET_SIZE: int     = 2630
    else:
        raise ValueError('set_type must be either "train" or "test"')

    IM_SIZE: int            = 64
    NUM_CHANNELS: int       = 3
    NUM_CLASSES: int        = 43

    SHAPE_DIM: int          = 2
    COLOR_DIM: int          = 3
    SYMBOL_DIM: int         = 15

    if set_type == 'train':
        paths_dict: dict = {
            'train_data_path': 'local/memmap_data/signs_dom_trn_data.npy',
            'train_labels_path': f'local/memmap_data/signs_dom_trn_labels.npy',
            'valid_data_path': 'local/memmap_data/signs_csl_trn_data.npy',
            'valid_labels_path': f'local/memmap_data/signs_csl_trn_labels.npy'
        }
    elif set_type == 'test':
        paths_dict: dict = {
            'train_data_path': 'local/memmap_data/signs_dom_trn_data.npy',
            'train_labels_path': f'local/memmap_data/signs_dom_trn_labels.npy',
            'valid_data_path': 'local/memmap_data/signs_tst_data.npy',
            'valid_labels_path': f'local/memmap_data/signs_tst_labels.npy'
        }
    else:
        raise ValueError('set_type must be either "train" or "test"')

    shapes_dict: dict = {
        'train_data_shape': (TRAIN_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'train_labels_shape': (TRAIN_SET_SIZE, NUM_CLASSES),
        'valid_data_shape': (VALID_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'valid_labels_shape': (VALID_SET_SIZE, NUM_CLASSES)
    }

    BATCH_SIZE: int         = 256

    ctrl = ImageLearningController(learner_type='domain_learner')
    ctrl.create_data_loaders(
        dataset='local',
        batch_size=BATCH_SIZE,
        paths_dict=paths_dict,
        shapes_dict=shapes_dict
    )

    # --- create the directory if it doesn't exist ---

    if not os.path.exists('local/causal_data'):
        os.makedirs('local/causal_data')

    # --- shapes domain ---

    ctrl.load_pretrained_learner(learner_path='local/models/shapes_model/')

    features = np.empty((0,SHAPE_DIM))
    labels = np.empty((0,NUM_CLASSES))
    total = 0
    for batch_data, batch_labels in ctrl.validation_loader:
        batch_features = ctrl.encoder.predict(batch_data, verbose=0)
        batch_features = layers.ReparameterizationLayer(SHAPE_DIM)(batch_features)[2]
        batch_features = batch_features.numpy()
        features = np.append(features, batch_features, axis=0)
        labels = np.append(labels, batch_labels[1], axis=0)
        total += len(batch_labels[1])
        print(f'\rComputing shape features: {total}/{VALID_SET_SIZE}', end='')
    print('\nDone')

    df['label'] = np.argmax(labels, axis=1)

    columns = {f's{i}': features[:,i] for i in range(SHAPE_DIM)}
    df = pd.concat([df, pd.DataFrame(columns)], axis=1)

    K.clear_session()

    # --- colors domain ---

    ctrl.load_pretrained_learner(learner_path='local/models/colors_model/')

    features = np.empty((0,COLOR_DIM))
    total = 0
    for batch_data, batch_labels in ctrl.validation_loader:
        batch_features = ctrl.encoder.predict(batch_data, verbose=0)
        batch_features = layers.ReparameterizationLayer(COLOR_DIM)(batch_features)[2]
        batch_features = batch_features.numpy()
        features = np.append(features, batch_features, axis=0)
        total += len(batch_labels[1])
        print(f'\rComputing color features: {total}/{VALID_SET_SIZE}', end='')
    print('\nDone')

    columns = {f'c{i}': features[:,i] for i in range(COLOR_DIM)}
    df = pd.concat([df, pd.DataFrame(columns)], axis=1)

    K.clear_session()

    # --- symbols domain ---

    ctrl.load_pretrained_learner(learner_path='local/models/symbols_model/')

    features = np.empty((0,SYMBOL_DIM))
    total = 0
    for batch_data, batch_labels in ctrl.validation_loader:
        batch_features = ctrl.encoder.predict(batch_data, verbose=0)
        batch_features = layers.ReparameterizationLayer(SYMBOL_DIM)(batch_features)[2]
        batch_features = batch_features.numpy()
        features = np.append(features, batch_features, axis=0)
        total += len(batch_labels[1])
        print(f'\rComputing symbol features: {total}/{VALID_SET_SIZE}', end='')
    print('\nDone')

    columns = {f'y{i}': features[:,i] for i in range(SYMBOL_DIM)}
    df = pd.concat([df, pd.DataFrame(columns)], axis=1)

    if set_type == 'train':
        df.to_csv('local/causal_data/sign_cs_trn.csv', index=False)
    elif set_type == 'test':
        df.to_csv('local/causal_data/sign_cs_tst.csv', index=False)
    else:
        raise ValueError('set_type must be either "train" or "test"')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    sets = ['train', 'test']

    for set_type in sets:
        main(set_type)
        K.clear_session()

# ------------------------------------------------------------------------------