"""
This module contains code for the linear block code used in the effective
communication system to increase the bit rate to match the other systems.

The code is a simple (n, 2) linear block code where
- The first codeword is all zeros
- The second codeword starts with 0 and alternates 0 and 1
- The third codeword starts with 1 and alternates 0 and 1
- The fourth codeword is all ones

The code had a minimum Hamming distance of n/2 and can correct up to n/2 - 1
errors (for n even).
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np

# ------------------------------------------------------------------------------

class M2LinearBlockCode():
    """
    This class defines a simple linear block code that can encode and decode
    messages, where the code is a (n, 2) code.

    The code is a simple (n, 2) linear block code where
    - The first codeword is all zeros
    - The second codeword starts with 0 and alternates 0 and 1
    - The third codeword starts with 1 and alternates 0 and 1
    - The fourth codeword is all ones

    The code had a minimum Hamming distance of n/2 and can correct up to n/2 - 1
    errors.
    """
    def __init__(self, codeword_length: int):
        """
        Initialize the code with the codeword length.

        Parameters:
        - codeword_length: int
            The length of the codewords in the code.
        """
        self.n = codeword_length

        cw0 = np.zeros((self.n,), dtype=int)
        cw1 = cw0.copy(); cw1[1::2] = 1
        cw2 = cw0.copy(); cw2[::2] = 1
        cw3 = np.ones((self.n,), dtype=int)
        self.codebook = {0: cw0, 1: cw1, 2: cw2, 3: cw3}

        self.codebook_mat = np.array([cw0, cw1, cw2, cw3]).T

    def encode(self, message: int) -> np.ndarray:
        """
        Encode a message using the code.

        Parameters
        ---------
        message : int
            The message to encode. Must be in {0, 1, 2, 3}.

        Returns
        -------
        codeword: np.ndarray
            The codeword corresponding to the message, as a numpy array of bits.
        """
        return self.codebook[message]
    
    def decode(self, received: np.ndarray) -> int:
        """
        Decode a received codeword.

        Parameters
        ---------
        received : np.ndarray
            The received codeword to decode, as a 1D numpy array of bits.

        Returns
        -------
        message: int
            The message decoded from the received codeword. One of {0, 1, 2, 3}.
        """
        reshaped_bits = np.reshape(received, (self.n, 1))
        errors = np.bitwise_xor(self.codebook_mat, reshaped_bits)
        distances = np.sum(errors, axis=0)
        return np.argmin(distances)