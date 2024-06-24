"""
This module contains code for converting between IEEE754 32-bit floating point
numbers and their binary representations.

Resources references when created the IEEE754 functions:
- primary reference: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_math.html
- wikipedia: https://en.wikipedia.org/wiki/IEEE_754
- converter for checking: https://www.h-schmidt.net/FloatConverter/IEEE754.html
"""

# ------------------------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------------------------

def ieee754_b2f(bits: np.ndarray) -> float:
    """
    This function converts a 32-bit binary number to a float according to the
    IEEE754 standard.

    Parameters
    ----------
    bits : np.ndarray
        A 1D numpy array of 32 bits.

    Returns
    -------
    float
        The float value represented by the 32 bits.
    """
    if len(bits) != 32:
        raise ValueError('bits should have length 32')

    s = bits[0]
    e = bits[1:9]
    f = bits[9:]

    exponent_weights = 2 ** np.arange(7, -1, -1)
    e_int = np.dot(e, exponent_weights)

    fraction_weights = 2.0 ** np.arange(-1, -24, -1)
    f_float = np.dot(f, fraction_weights)

    if s == 0 and e_int == 255 and f_float == 0:
        return np.inf
    elif s == 1 and e_int == 255 and f_float == 0:
        return -np.inf
    elif e_int == 255 and f_float != 0:
        return np.nan
    elif e_int == 0 and f_float == 0:
        return (-1)**s * 0.0
    elif e_int == 0 and f_float != 0:
        return (-1)**s * 2.0**(-126) * f_float
    else:
        return (-1)**s * 2.0**(e_int-127) * (1+f_float)
    
# ------------------------------------------------------------------------------

def ieee754_f2b(float_val: float) -> np.ndarray:
    """
    This function converts a float to a 32-bit binary number according to the
    IEEE754 standard.

    Parameters
    ----------
    float_val : float
        The float value to convert.

    Returns
    -------
    np.ndarray
        A 1D numpy array of 32 bits.
    """
    max_norm    = 3.4028234663852886e+38
    min_norm    = 1.1754943508222875e-38
    max_subnorm = 1.1754942106924411e-38
    min_subnorm = 1.401298464324817e-45

    if float_val == 0.0:
        return _hex_to_bits('00000000')
    elif np.isnan(float_val):
        return _hex_to_bits('7fc00000')
    elif float_val == np.inf:
        return _hex_to_bits('7f800000')
    elif float_val == -np.inf:
        return _hex_to_bits('ff800000')
    elif abs(float_val) > max_norm:
        raise ValueError('non-inf float_val is too large')
    elif abs(float_val) < min_subnorm:
        raise ValueError('non-inf float_val is too small')
    elif np.abs(float_val) < min_norm:
        # subnormal logic
        s = 0 if float_val > 0 else 1
        e = np.zeros(8).astype(np.uint8)
        f = _d2b(float_val*2**126, 23)
        return np.concatenate([np.array([s]), e, f])
    else:
        # normal logic
        ncalcbits = 128
        s = 0 if float_val > 0 else 1
        abs_float = np.abs(float_val)
        whole_part = int(abs_float)
        frac_part = abs_float - whole_part
        whole_bin = _d2b(whole_part, ncalcbits)
        frac_bin = _d2b(frac_part, ncalcbits)
        if np.sum(whole_bin) == 0:
            # move decimal right
            first_one = np.argmax(frac_bin)
            moves = first_one + 1
            exp   = int(127-moves)
            e     = _d2b(exp, 8)
            # build the mantissa
            f     = frac_bin[first_one+1:]
            if len(f) < 23:
                f = np.concatenate([f, np.zeros(23-len(f)).astype(np.uint8)])
            if len(f) > 23:
                f = f[:23]
            return np.concatenate([np.array([s]), e, f])
        else:
            # move decimal left
            first_one = np.argmax(whole_bin)
            moves = ncalcbits-1-first_one
            exp   = int(127+moves)
            e     = _d2b(exp, 8)
            # build the mantissa
            f     = np.concatenate([whole_bin[first_one+1:], frac_bin])
            if len(f) < 23:
                f = np.concatenate([f, np.zeros(23-len(f)).astype(np.uint8)])
            if len(f) > 23:
                f = f[:23]
            return np.concatenate([np.array([s]), e, f])
        
# ------------------------------------------------------------------------------

def _hex_to_bits(hex_str: str) -> np.ndarray:
    """
    This function takes a hexadecimal string and converts it to a 32-bit binary
    numpy array.
    """
    return np.array([int(b) for b in bin(int(hex_str, 16))[2:].zfill(32)])

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

if __name__ == '__main__':

    print(ieee754_f2b(-1.0))
    print(ieee754_f2b(-0.5))
    print(ieee754_f2b(0.0))
    print(ieee754_f2b(0.5))
    print(ieee754_f2b(1.0))