"""
This module contains classes for simulating wireless channels.
"""

# ------------------------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------------------------

class AWGN():
    """
    This class will simulate an AWGN channel.
    """
    def __init__(self, standard_dev=1):
        """
        Constructor for the AWGN class.

        Parameters
        ----------
        standard_dev : float, optional
            The standard deviation of the AWGN. The default is 1.
        """
        self.std = standard_dev
        
    def add_real_noise(self, signal):
        """
        This method adds AWGN to a real signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The real signal to which to add noise.

        Returns
        -------
        numpy.ndarray
            The signal with added noise.
        """
        signal = signal + self.std*np.random.normal(size=np.shape(signal))
        return signal
    
    def add_complex_noise(self, signal):
        """
        This method adds complex AWGN to a complex signal.
        
        Parameters
        ----------
        signal : numpy.ndarray
            The complex signal to which to add noise.

        Returns
        -------
        numpy.ndarray
            The signal with added noise.
        """
        n_r     = self.std/np.sqrt(2)*np.random.normal(size=np.shape(signal))
        n_i     = self.std/np.sqrt(2)*np.random.normal(size=np.shape(signal))
        noise   = n_r + 1j*n_i
        return signal + noise
    
# ------------------------------------------------------------------------------

class Rayleigh(AWGN):
    """
    This class will simulate a Rayleigh fading channel. The fading coefficients
    differ over each symbol, so the symbol period is approximately equal to the
    channel coherence time (flat fading over each symbol).

    This class inherits from the AWGN class, as the additive noise is assumed
    to be Gaussian.
    """
    def __init__(self, scale=1/np.sqrt(2), awgn_std=1):
        """
        Constructor for the Rayleigh class.

        Parameters
        ----------
        scale : float, optional
            The scale parameter of the Rayleigh distribution. The default is 
            1/sqrt(2) (standard Rayleigh fading with average power of 1).
        awgn_std : float, optional
            The standard deviation of the AWGN. The default is 1.
        """
        super(Rayleigh, self).__init__(awgn_std)
        self.scale  = scale
        self.std    = awgn_std
    
    def fade_signal(self, signal):
        """
        This method applies Rayleigh fading to the modulated signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The modulated signal.

        Returns
        -------
        numpy.ndarray
            The signal with Rayleigh fading applied.
        numpy.ndarray
            The fading coefficients. This are returned so that equalization
            can be performed at the receiver.
        """
        real    = np.random.normal(size=np.shape(signal), scale=self.scale)
        imag    = np.random.normal(size=np.shape(signal), scale=self.scale)
        coefs   = real + 1j*imag
        faded   = np.multiply(signal, coefs)
        return faded, coefs
    
# ------------------------------------------------------------------------------

class Rician(AWGN):
    """
    This class will simulate a Rician fading channel. The fading coefficients
    differ over each symbol, so the symbol period is approximately equal to the
    channel coherence time.

    This class inherits from the AWGN class, as the additive noise is assumed
    to be Gaussian.
    """
    def __init__(self, scale=1/np.sqrt(2), noncentral=0, awgn_std=1):
        """
        Constructor for the Rician class.

        Parameters
        ----------
        scale : float, optional
            The scale parameter of the Rician distribution. The default is 1.
        noncentral : float, optional
            The noncentrality parameter of the Rician distribution. The default
            is 0.
        awgn_std : float, optional
            The standard deviation of the AWGN. The default is 1.
        """
        super(Rician, self).__init__(awgn_std)
        self.sig    = scale
        self.s      = noncentral
        self.std    = awgn_std
    
    # to test fading performance, use technique from pg 189 of Goldsmith
    def fade_signal(self, signal):
        """
        This method applies Rician fading to the modulated signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The modulated signal.

        Returns
        -------
        numpy.ndarray
            The signal with Rician fading applied.
        numpy.ndarray
            The fading coefficients. These are returned so that equalization
            can be performed at the receiver.
        """
        # compute fading parameter K
        K       = (self.s**2)/2/(self.sig**2)
        # generate the coefficients
        shape       = np.shape(signal)
        poiss       = np.random.poisson(lam=K, size=shape)
        chisq       = np.zeros(shape)
        for i in range(len(poiss)):
            chisq[i]    = np.random.chisquare(df=2*poiss[i]+2)
        coefs       = self.sig*np.sqrt(chisq)
        # apply the coeffcients to each symbol
        faded   = np.multiply(signal, coefs)
        return faded, coefs
        
# ------------------------------------------------------------------------------