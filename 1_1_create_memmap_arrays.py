"""
This script uses the pipeline module to load the raw data and save it as
memory-mapped NumPy arrays.
"""

# ------------------------------------------------------------------------------
# imports

import os
import numpy as np
import modules.pipeline as pipe
import modules.visualize_data as vis

# ------------------------------------------------------------------------------

def main():

    # --- setup ---

    SIZE = 64
    TEST_NUMBER = 2630

    # directories
    trn_image_dir = 'local/raw_data/train/images/'
    trn_anno_dir = 'local/raw_data/train/anno/'
    tst_image_dir = 'local/raw_data/test/images/'
    tst_anno_file = 'local/raw_data/test/anno.csv'
    save_dir = 'local/memmap_data'

    # --- create the save directory if it does not exist ---

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Save directory created: {save_dir}')
    else:
        print(f'Save directory already exists: {save_dir}')

    # --- training data ---
    
    # create empty arrays for the training data
    train_data = np.empty((0, SIZE, SIZE, 3), dtype=np.float32)
    train_labels = np.empty(0, dtype=np.int64)

    # get the training directories/files and sort
    data_dirs = os.listdir(trn_image_dir)
    data_dirs = list(np.sort(data_dirs))
    label_files = os.listdir(trn_anno_dir)
    label_files = list(np.sort(label_files))

    # load all data and labels
    for i in range(len(data_dirs)):
        data_dir = data_dirs[i]
        label_file = label_files[i]

        dir_ims = pipe.load_images(trn_image_dir + data_dir + '/', SIZE, 
                                   verbose=False)
        dir_labels = pipe.load_labels(trn_anno_dir + label_file)

        train_data = np.append(train_data, dir_ims, axis=0)
        train_labels = np.append(train_labels, dir_labels)

        print(
            f'\rLoading training data and labels: {i+1}/{len(data_dirs)}', 
            end=''
        )
    print('\nDone.')

    train_labels = pipe.labels_to_one_hot(train_labels)
    train_data, train_labels = pipe.shuffle_data(train_data, train_labels)

    # # UNCOMMENT TO VISUALIZE TRAIN DATA
    # _labels = np.argmax(train_labels, axis=1)
    # vis.show_processed_images(train_data, _labels, 4)

    print('\nTraining data and labels loaded.')
    print(f'train_data.shape: {train_data.shape}')
    print(f'train_labels.shape: {train_labels.shape}')

    # --- test data ---

    # load the test data
    test_data = pipe.load_images(tst_image_dir, SIZE, verbose=True)
    test_labels = pipe.load_labels(tst_anno_file)
    test_labels = pipe.labels_to_one_hot(test_labels)
    test_data, test_labels = pipe.shuffle_data(test_data, test_labels)

    # # UNCOMMENT TO VISUALIZE TEST DATA
    # _labels = np.argmax(test_labels, axis=1)
    # vis.show_processed_images(test_data, _labels, 4)

    print('\nTest data and labels loaded.')
    print(f'test_data.shape: {test_data.shape}')
    print(f'test_labels.shape: {test_labels.shape}')

    # --- save data ---
    pipe.save_data(save_dir, train_data, train_labels, test_data, test_labels,
                   TEST_NUMBER)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------