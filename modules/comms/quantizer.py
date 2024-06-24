"""
This module contains the quantizer object for converting continuous values to
bits and vice versa.

The quantizer implemented here is a simple uniform quantizer with a fixed
number of bits and a fixed range.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np

# ------------------------------------------------------------------------------

class Quantizer():
    """
    This class defines a quantizer object for converting continuous values to
    bits and vice versa.
    """
    def __init__(self, n_bits: int, q_range: tuple):
        """
        Constructor for the Quantizer class.

        Parameters
        ----------
        n_bits : int
            The number of bits to represent each value.
        q_range : tuple
            The range of the quantizer.
        """
        self.n_bits = n_bits
        self.q_range = q_range
        self.q_step = (q_range[1] - q_range[0]) / (2**n_bits - 1)
        self.exps = np.flip(2**np.arange(n_bits).astype(np.float64))

    def quantize(self, value: float):
        """
        Quantize the given values.

        Parameters
        ----------
        value : float
            The value to quantize.

        Returns
        -------
        np.ndarray
            A 1D numpy array of bits.
        """
        if value < self.q_range[0]:
            value = self.q_range[0]
        elif value > self.q_range[1]:
            value = self.q_range[1]

        q_value = (value - self.q_range[0]) / self.q_step
        q_value = round(q_value)
        bits = _d2b(q_value, self.n_bits)
        return bits

    def dequantize(self, bits: np.ndarray):
        """
        Dequantize the given bits.

        Parameters
        ----------
        bits : np.ndarray
            A 1D numpy array of bits.

        Returns
        -------
        float
            The dequantized value.
        """
        if len(bits) != self.n_bits:
            raise ValueError('bits should have length n_bits')
        q_value = np.dot(bits, self.exps)
        deq_value = q_value * self.q_step + self.q_range[0]
        return deq_value

# ------------------------------------------------------------------------------

def _d2b(decimal, nbits) -> np.ndarray:
    """
    This function takes a decimal number and converts it to a specified number
    of bits in the form of a numpy array.
    """
    bits = np.zeros(nbits).astype(np.uint8)
    if decimal < 1.0:
        # fraction logic
        exps = 0.5**np.arange(1, nbits+1)
        for i in range(1, nbits+1):
            if decimal >= exps[i-1]:
                decimal -= exps[i-1]
                bits[i-1] = 1
    else:
        if not isinstance(decimal, int):
            raise ValueError('decimal should be an integer if >= 1.0')
        else:
            # integer logic
            exps = 2**np.arange(nbits).astype(np.float64)
            for i in range(nbits):
                if decimal >= exps[nbits-1-i]:
                    decimal -= exps[nbits-1-i]
                    bits[i] = 1
    return bits

# ------------------------------------------------------------------------------

def _b2d(bits) -> int:
    """
    This function takes a numpy array of bits and converts it to a decimal
    number. Assumes that the bits represent an unsigned integer.
    """
    decimal = np.dot(bits, 2**np.arange(len(bits))[::-1])
    return decimal

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n_bits = 10
    q_range = (-10.0, 10.0)

    q = Quantizer(n_bits, q_range)

    values = np.linspace(1.1*q_range[0], 1.1*q_range[1], 10000)
    recovered = np.zeros_like(values)
    for i, value in enumerate(values):
        bits = q.quantize(value)
        recovered[i] = q.dequantize(bits)
    
    plt.figure(figsize=(8, 6))
    plt.plot(values, recovered, color='blue', label='Quantizer')
    plt.plot(values, values, color='red', label='Ideal')
    plt.xlabel('Original Value')
    plt.xlim((-1.2, 1.2))
    plt.ylim((-1.2, 1.2))
    plt.ylabel('Recovered Value')
    plt.legend()
    plt.grid()
    plt.show()
