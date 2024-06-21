"""
This module contains the reasoning module for the semantic communication with 
conceptual spaces system.

As a first iteration, this module will take in
- the treatment matrix
- the conceptual space features
- the maximum message length (in terms of CS features)
and it will output the best message, where the features are selected based on
their overall effect on the task variables.

In this first iteration, the overall effect is computed as the row-sum of the
absolute values of the treatment matrix.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np

# ------------------------------------------------------------------------------

class Reasoner():
    """
    This class defines the reasoning module for the SCCS system.
    """

    def __init__(self, effect_matrix: np.ndarray):
        """
        Constructor for the Reasoner class.

        Parameters
        ----------
        effect_matrix : np.ndarray
            The treatment effect matrix. Entries are the average treatment
            effect of the corresponding treatment variables (rows) on the task
            variables (columns).
        """
        self.effect_matrix = effect_matrix
        
        num_matrix = np.nan_to_num(effect_matrix)
        abs_matrix = np.abs(num_matrix)
        row_sums = np.sum(abs_matrix, axis=1).ravel()
        order = np.argsort(row_sums)
        self.priority = np.argsort(order)

    def decide(self, qualities: np.ndarray, max_length: int):
        """
        Decide which values of the quality dimensions to communicate. Dimensions
        that are not communicated are masked with NaN.

        Parameters
        ----------
        qualities : np.ndarray
            The quality dimensions of the conceptual space.
        max_length : int
            The maximum number of quality values to communicate.
        """
        if len(qualities.shape) > 1 or not isinstance(qualities, np.ndarray):
            raise ValueError('qualities should be a 1D numpy array')
        if max_length <= 0 or not isinstance(max_length, int):
            raise ValueError('max_length should be a positive integer')
        
        threshold = len(qualities) - max_length
        
        if max_length > len(qualities):
            return qualities
        else:
            sent = qualities.copy()
            sent[self.priority < threshold] = np.nan
            return sent

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    from .matrices import sc_colors_full

    R = Reasoner(sc_colors_full)

    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    max_length = 3

    print(R.decide(data, max_length))

