"""
This script impelements the pipeline for the data, assigning new labels
according to the chosen map.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import modules.pipeline as pipe
from modules.label_maps import maps

# ------------------------------------------------------------------------------

def main(identifier):

    label_map = maps[identifier]

    dom_trn_size = 39209
    csl_trn_size = 10000
    test_size = 2630

    num_classes = 43

    dom_trn_shape = (dom_trn_size, num_classes)
    csl_trn_shape = (csl_trn_size, num_classes)
    tst_shape = (test_size, num_classes)

    dom_trn_file = 'local/memmap_data/signs_dom_trn_labels.npy'
    csl_trn_file = 'local/memmap_data/signs_csl_trn_labels.npy'
    test_file = 'local/memmap_data/signs_tst_labels.npy'
    
    mm_dom_trn_labels = pipe.load_memmap_labels(dom_trn_file, dom_trn_shape)
    dom_trn_labels = np.argmax(mm_dom_trn_labels, axis=1)

    mm_csl_vld_labels = pipe.load_memmap_labels(csl_trn_file, csl_trn_shape)
    csl_trn_labels = np.argmax(mm_csl_vld_labels, axis=1)

    mm_tst_labels = pipe.load_memmap_labels(test_file, tst_shape)
    tst_labels = np.argmax(mm_tst_labels, axis=1)

    new_dom_trn_labels = pipe.map_labels(dom_trn_labels, label_map)
    new_csl_trn_labels = pipe.map_labels(csl_trn_labels, label_map)
    new_tst_labels = pipe.map_labels(tst_labels, label_map)

    new_dom_trn_labels = pipe.labels_to_one_hot(new_dom_trn_labels)
    new_csl_trn_labels = pipe.labels_to_one_hot(new_csl_trn_labels)
    new_tst_labels = pipe.labels_to_one_hot(new_tst_labels)

    pipe.save_new_labels(
        'local/memmap_data/',
        new_dom_trn_labels,
        new_csl_trn_labels,
        new_tst_labels,
        identifier
    )

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main('shapes')
    main('colors')

# ------------------------------------------------------------------------------