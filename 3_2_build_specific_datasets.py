"""
This script will build the specific datasets over which to evaluate the causal
reasoning framework.

The generated training/test datasets are of the following combinations:
- shape task, shape/color data
- color task, shape/color data
- isSpeedLimit task, shape/color/symbol data
"""

# ------------------------------------------------------------------------------
# imports

import pandas as pd
import numpy as np

from modules.label_maps import maps

# ------------------------------------------------------------------------------

def main(which_data: list, which_label: str, set_type: str):

    SHAPE_DIM = 2
    COLOR_DIM = 3
    SYMBOL_DIM = 15

    if set_type == 'train':
        NUM_POINTS = 10000
        TYPE_ID = 'trn'
    elif set_type == 'test':
        NUM_POINTS = 2630
        TYPE_ID = 'tst'

    DATA_DIR = 'local/causal_data/'
    DATA_FILE = f'sign_cs_{TYPE_ID}.csv'
    DATA_PATH = DATA_DIR + DATA_FILE

    if which_label == 'isSpeedLimit':
        NUM_CLASSES = 2
        MAP = maps['isSpeedLimit']
    elif which_label == 'shapes':
        NUM_CLASSES = 4
        MAP = maps['shapes']
    elif which_label == 'colors':
        NUM_CLASSES = 4
        MAP = maps['colors']
    else:
        raise ValueError('Invalid label type')

    if NUM_CLASSES > 2:
        LABEL_SHAPE = (NUM_POINTS, NUM_CLASSES)
    else:
        LABEL_SHAPE = (NUM_POINTS,)

    # --- load data ---

    data = pd.read_csv(DATA_PATH)

    # map labels

    original_labels = data['label'].values
    new_labels = np.array([MAP[l] for l in original_labels])
    labels = np.zeros(LABEL_SHAPE)
    if NUM_CLASSES > 2:
        labels[np.arange(NUM_POINTS), new_labels] = 1
    else:
        labels = new_labels

    # --- build dataset ---

    df = pd.DataFrame()

    if 'shapes' in which_data:
        columns = {f's{i}': data[f's{i}'] for i in range(SHAPE_DIM)}
        df = pd.concat([df, pd.DataFrame(columns)], axis=1)
    if 'colors' in which_data:
        columns = {f'c{i}': data[f'c{i}'] for i in range(COLOR_DIM)}
        df = pd.concat([df, pd.DataFrame(columns)], axis=1)
    if 'symbols' in which_data:
        columns = {f'y{i}': data[f'y{i}'] for i in range(SYMBOL_DIM)}
        df = pd.concat([df, pd.DataFrame(columns)], axis=1)

    if NUM_CLASSES > 2:
        columns = {f'u{i}' : labels[:,i] for i in range(NUM_CLASSES)}
    else:
        columns = {'u': labels}
    df = pd.concat([df, pd.DataFrame(columns)], axis=1)

    # --- save dataset ---

    filename = f'{TYPE_ID}_'
    for i, d in enumerate(which_data):
        filename += d + '-' if i < len(which_data) - 1 else d
    filename += f'_{which_label}'
    filename += '.csv'

    df.to_csv(DATA_DIR + filename, index=False)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    SET_TYPES = ['train', 'test']

    WHICH_LABELS1 = ['shapes', 'colors']
    WHICH_DATA1 = ['shapes', 'colors']

    WHICH_LABELS2 = ['isSpeedLimit']
    WHICH_DATA2 = ['shapes', 'colors', 'symbols']

    for set_type in SET_TYPES:
        for which_label in WHICH_LABELS1:
            main(WHICH_DATA1, which_label, set_type)

        for which_label in WHICH_LABELS2:
            main(WHICH_DATA2, which_label, set_type)

# ------------------------------------------------------------------------------