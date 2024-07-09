"""
This script will evaluate an already trained domain learner.
"""

# ------------------------------------------------------------------------------
# imports

import os
import numpy as np
import keras.backend as K

from modules.cslearn.controllers import ImageLearningController

# ------------------------------------------------------------------------------

def main(domain: str):

    figpath = 'local/figures/encoders/'

    if not os.path.exists(figpath):
        os.makedirs(figpath)

    TRAIN_SET_SIZE: int     = 39209
    VALID_SET_SIZE: int     = 2630

    IM_SIZE: int            = 64
    NUM_CHANNELS: int       = 3
    NUM_CLASSES: int        = 43 if domain == 'symbols' else 4

    BATCH_SIZE: int         = 64

    DATA_DIR: str          = 'local/memmap_data/'

    if domain is not 'symbols':
        PATHS_DICT: dict = {
            'train_data_path': DATA_DIR + 'signs_dom_trn_data.npy',
            'train_labels_path': DATA_DIR + f'signs_dom_trn_labels_{domain}.npy',
            'valid_data_path': DATA_DIR + 'signs_tst_data.npy',
            'valid_labels_path': DATA_DIR + f'signs_tst_labels_{domain}.npy'
        }
    else:
        PATHS_DICT: dict = {
            'train_data_path': DATA_DIR + 'signs_dom_trn_data.npy',
            'train_labels_path': DATA_DIR + f'signs_dom_trn_labels.npy',
            'valid_data_path': DATA_DIR + 'signs_tst_data.npy',
            'valid_labels_path': DATA_DIR + f'signs_tst_labels.npy'
        }

    SHAPES_DICT: dict = {
        'train_data_shape': (TRAIN_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'train_labels_shape': (TRAIN_SET_SIZE, NUM_CLASSES),
        'valid_data_shape': (VALID_SET_SIZE, IM_SIZE, IM_SIZE, NUM_CHANNELS),
        'valid_labels_shape': (VALID_SET_SIZE, NUM_CLASSES)
    }

    ctrl = ImageLearningController(learner_type='domain_learner')

    ctrl.create_data_loaders(dataset='local', batch_size=BATCH_SIZE,
                             paths_dict=PATHS_DICT, shapes_dict=SHAPES_DICT)
    
    ctrl.load_pretrained_learner(learner_path=f'local/models/{domain}_model/')
    training_accuracy = np.load(f'local/models/{domain}_model/train_acc.npy')
    validation_accuracy = np.load(f'local/models/{domain}_model/valid_acc.npy')
    ctrl.training_history = {}
    ctrl.training_history['accuracy'] = list(training_accuracy)
    ctrl.training_history['val_accuracy'] = list(validation_accuracy)
    ctrl.models_trained = True

    ctrl.model.summary()

    # # --- evaluation ---

    if domain == 'shapes':
        legend = ['triangle', 'square', 'octagon', 'circle']
    elif domain == 'colors':
        legend = ['red', 'blue', 'yellow', 'black']
        colors = ['#FF0000', '#0000FF', '#DDDD00', '#000000']
    elif domain == 'symbols':
        legend = [str(i) for i in range(43)]
    else:
        raise ValueError('Invalid domain.')

    ctrl.eval_plot_accuracy_curves(save_path=f'{figpath}{domain}_acc.pdf')

    markers = ['^', 's', '*', 'o']

    if domain is not 'symbols':
        ctrl.eval_plot_scattered_features(
            legend=legend,
            save_path=f'{figpath}{domain}_scatteredFeats.pdf',
            colors=colors if domain == 'colors' else None,
            markers=markers if domain == 'shapes' else None
        )

    ctrl.eval_plot_similarity_heatmap(legend=legend,
                                      save_path=f'{figpath}{domain}_simHeatmap.pdf')

    ctrl.eval_show_decoded_protos(
        legend=legend,
        save_path=f'{figpath}{domain}_dec_protos.pdf'
    )

    # ctrl.eval_compare_true_and_generated(which='training')

    # ctrl.eval_compare_true_and_generated(
    #     which='validation',
    #     save_path=f'{figpath}{domain}_trueVsGen.pdf'
    # )

    if domain == 'shapes':
        fixed_dims = [0.0, 0.0]
    elif domain == 'colors':
        fixed_dims = [0.0, 0.0, 0.0]
    elif domain == 'symbols':
        fixed_dims = [0.0] * 15
    else:
        raise ValueError('Invalid domain.')

    if domain is 'shapes':
        ctrl.eval_visualize_all_dimensions(
            save_path=f'{figpath}{domain}_visDims.pdf', fixed_dims=fixed_dims,
            figsize=(7.16,1.45), steps=10)
    elif domain is 'colors':
        ctrl.eval_visualize_all_dimensions(
            save_path=f'{figpath}{domain}_visDims.pdf', fixed_dims=fixed_dims,
            figsize=(7.16,2.15), steps=10)
    elif domain is 'symbols':
        ctrl.eval_visualize_all_dimensions(
            save_path=f'{figpath}{domain}_visDims.pdf', fixed_dims=fixed_dims,
            figsize=(7.16,10.8), steps=10)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    domains = ['shapes']

    for domain in domains:
        print(f'\n\nEvaluating domain learner on {domain} data.')
        main(domain)
        K.clear_session()

# ------------------------------------------------------------------------------