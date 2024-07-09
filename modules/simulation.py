"""
This module contains the simulation functions for the various systems.
"""

# ------------------------------------------------------------------------------
# imports

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt


from modules.comms.wifi_phy import PHY
from modules.comms.channels import AWGN, Rayleigh
from modules.comms.quantizer import Quantizer
from modules.comms.linear_block_code import M2LinearBlockCode

from modules.causal_cs.reasoning import Reasoner
from modules.causal_cs import matrices

# ------------------------------------------------------------------------------

def simulate_sccsr(effect_matrix: np.ndarray, number_sims: int = 500,
                   snrs: np.ndarray = np.linspace(-20, 20, 7),
                   reasoning: bool = True, message_length: int = 2,
                   task: str = 'shapes', modulation_type: str = 'BPSK',
                   channel_coding: bool = False, code_rate: str = '1/2',
                   vitdec_type: str = 'soft', quant_bits: int = 32,
                   quant_range: tuple = (-5, 5), channel: str = 'awgn'
                ):
    """
    This function simulates the SCCS-R system.

    Parameters
    ----------
    effect_matrix : np.ndarray
        The effect matrix for the system.
    number_sims : int
        The number of simulated messages to run for each trial.
        Default is 500.
    snrs : np.ndarray
        The SNR values to simulate.
        Default is np.linspace(-20, 20, 7).
    reasoning : bool
        Whether to use the reasoner or not.
        Default is True.
    message_length : int
        The length of the message to send.
        Default is 2.
    task : str
        The task to simulate.
        Default is 'shapes'.
    modulation_type : str
        The modulation type to use.
        Default is 'BPSK'.
    channel_coding : bool
        Whether to use channel coding or not.
        Default is False.
    code_rate : str
        The code rate to use. Only used if channel_coding is True.
        Default is '1/2'.
    vitdec_type : str
        The type of Viterbi decoder to use. Only used if channel_coding is True.\
        Default is 'soft'.
    quant_bits : int
        The number of bits to use for quantization.
        Default is 32.
    quant_range : tuple
        The range of values to quantize to.
        Default is (-5, 5).
    channel : str
        The channel to use. Options are 'awgn' and 'rayleigh'.
        Default is 'awgn'.
    """
    
    # --- constants ---

    DATA_DIR    = 'local/causal_data/'
    if task in ['shapes', 'colors']:
        DATA_FILE   = f'tst_shapes-colors_{task}.csv'
    elif task == 'isSpeedLimit':
        DATA_FILE   = f'tst_shapes-colors-symbols_{task}.csv'

    MODELS_DIR  = 'local/models/decoders/'
    MODEL_NAME  = f'{task}.keras'

    EB          = 1.0

    if channel == 'awgn':
        FADING = False
    elif channel == 'rayleigh':
        FADING = True
    else:
        raise ValueError(f'Invalid channel! Got {channel}')

    # --- load and format data ---

    df = pd.read_csv(DATA_DIR + DATA_FILE)

    if task in ['shapes', 'colors']:
        data = df[['s0', 's1', 'c0', 'c1', 'c2']].to_numpy().astype(np.float32)
        labels = df[['u0', 'u1', 'u2', 'u3']].to_numpy().astype(np.float32)
    elif task == 'isSpeedLimit':
        data = df[['s0', 's1', 'c0', 'c1', 'c2', 'y0', 'y1', 'y2', 'y3', 'y4',
                    'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13', 
                    'y14']].to_numpy().astype(np.float32)
        labels = df[['u']].to_numpy().astype(np.float32)
    else:
        raise ValueError(f'Invalid task! Got {task}')

    n_samples, n_dims = data.shape

    # --- instantiate the simulation objects ---

    phy = PHY(modulation_type, code_rate)
    qnt = Quantizer(quant_bits, quant_range)
    reasoner = Reasoner(effect_matrix)
    decoder = keras.models.load_model(MODELS_DIR + MODEL_NAME)

    # --- simulate the system ---

    bers = np.empty(snrs.shape, dtype=np.float32)
    accs = np.empty(snrs.shape, dtype=np.float32)

    sim_idx = np.random.choice(n_samples, number_sims, replace=False)

    for m, EBN0 in enumerate(snrs):

        N0 = EB / 10**(EBN0/10)
        std = np.sqrt(N0/2)
        if FADING:
            channel = Rayleigh(awgn_std=std)
        else:
            channel = AWGN(standard_dev=std)

        rx_samples = np.empty((number_sims, n_dims), dtype=np.float32)
        bit_errors = 0

        for k, sample in enumerate(data[sim_idx]):

            if reasoning:
                qualities = reasoner.decide(sample, max_length=message_length)
                nan_idx = np.where(np.isnan(qualities))[0]
            else:
                nan_idx = np.random.choice(n_dims, n_dims-message_length, replace=False)
                qualities = sample.copy()
                qualities[nan_idx] = np.nan
            
            # convert the message to bits (not NaN values)
            data_bits = np.empty((0,), dtype=np.uint8)
            for i, value in enumerate(sample):
                if i not in nan_idx:
                    val_bits = qnt.quantize(value)
                    data_bits = np.concatenate((data_bits, val_bits))

            # encode, modulate, transmit, receive, demodulate, decode
            if channel_coding:
                tx_bits = phy.conv_encode(data_bits)
            else:
                tx_bits = data_bits
            tx_symbols = phy.qam_modulate(tx_bits)
            if FADING:
                fade_symbols, fade_coefs = channel.fade_signal(tx_symbols)
            else:
                fade_symbols = tx_symbols
            if modulation_type == 'BPSK':
                rx_symbols = channel.add_real_noise(fade_symbols)
            else:
                rx_symbols = channel.add_complex_noise(fade_symbols)
            if FADING:
                eq_symbols = np.divide(rx_symbols, fade_coefs)
            else:
                eq_symbols = rx_symbols
            if channel_coding and vitdec_type == 'soft':
                rx_data_bits = phy.viterbi_decode(eq_symbols, dec_type='soft')
            elif channel_coding and vitdec_type == 'hard':
                rx_bits = phy.qam_demodulate(eq_symbols)
                rx_data_bits = phy.viterbi_decode(rx_bits, dec_type='hard')
            else:
                rx_data_bits = phy.qam_demodulate(eq_symbols)

            # convert the received bits to floats (filling in NaN values)
            rx_sample = np.empty((n_dims,), dtype=np.float32)
            j = 0
            for i in range(n_dims):
                if i in nan_idx:
                    rx_sample[i] = np.nan
                else:
                    rx_sample[i] = qnt.dequantize(
                        rx_data_bits[j*quant_bits:(j+1)*quant_bits]
                    )
                    j += 1

            # save off the received sample for later decoding
            rx_samples[k] = rx_sample

            # accumulate the results
            bit_errors += np.sum(np.abs(data_bits - rx_data_bits))

            print(f'\rSample {k+1}/{number_sims} processed.', end='')

        # decode the received samples
        print('\nDecoding...')
        decoded = decoder.predict(rx_samples, verbose=0)

        print(f'Finished EBN0 = {EBN0} dB.\n')

        if labels.shape[1] > 1:
            true = np.argmax(labels[sim_idx], axis=1)
            preds = np.argmax(decoded, axis=1)
        else:
            true = labels[sim_idx].flatten()
            preds = np.round(decoded.flatten())

        bers[m] = bit_errors / (number_sims * message_length * quant_bits)
        accs[m] = np.mean(preds == true)

    return accs, bers

# ------------------------------------------------------------------------------

def simulate_technical(number_sims: int = 500,
                       snrs: np.ndarray = np.linspace(-20, 20, 7),
                       task: str = 'shapes', modulation_type: str = 'BPSK',
                       channel_coding: bool = False, code_rate: str = '1/2',
                       vitdec_type: str = 'soft', quant_bits: int = 32,
                       quant_range: tuple = (-5, 5), channel: str = 'awgn'
                    ):
    """
    This function simulates the technical system.

    Parameters
    ----------
    number_sims : int
        The number of simulated messages to run for each trial.
        Default is 500.
    snrs : np.ndarray
        The SNR values to simulate.
        Default is np.linspace(-20, 20, 7).
    task : str
        The task to simulate.
        Default is 'shapes'.
    modulation_type : str
        The modulation type to use.
        Default is 'BPSK'.
    channel_coding : bool
        Whether to use channel coding or not.
        Default is False.
    code_rate : str
        The code rate to use. Only used if channel_coding is True.
        Default is '1/2'.
    vitdec_type : str
        The type of Viterbi decoder to use. Only used if channel_coding is True.
        Default is 'soft'.
    quant_bits : int
        The number of bits to use for quantization.
        Default is 32.
    quant_range : tuple
        The range of values to quantize to.
        Default is (-5, 5).
    channel : str
        The channel to use. Options are 'awgn' and 'rayleigh'.
        Default is 'awgn'.
    """

    # --- constants ---
    
    N_IMS = 2630
    IM_SIZE = 64
    N_CHN = 3
    if task in ['shapes', 'colors']:
        N_CLS = 4
    elif task == 'isSpeedLimit':
        N_CLS = 2

    model_path = f'local/models/classifiers/{task}.keras'
    images_path = 'local/memmap_data/signs_tst_data.npy'
    labels_path = f'local/memmap_data/signs_tst_labels_{task}.npy'

    EB          = 1.0

    if channel == 'awgn':
        FADING = False
    elif channel == 'rayleigh':
        FADING = True
    else:
        raise ValueError(f'Invalid channel! Got {channel}')

    # --- load data ---

    images = np.memmap(images_path, dtype=np.float32, mode='r', 
                       shape=(N_IMS, IM_SIZE, IM_SIZE, N_CHN))

    labels = np.memmap(labels_path, dtype=np.float32, mode='r', 
                       shape=(N_IMS, N_CLS))

    # --- instantiate components ---

    phy = PHY(modulation_type, code_rate)
    qnt = Quantizer(quant_bits, quant_range)
    classifier = keras.models.load_model(model_path)

    # --- simulate the system ---

    bers = np.empty(snrs.shape, dtype=np.float32)
    accs = np.empty(snrs.shape, dtype=np.float32)

    sim_idx = np.random.choice(N_IMS, number_sims, replace=False)

    for m, EBN0 in enumerate(snrs):

        N0 = EB / 10**(EBN0/10)
        std = np.sqrt(N0/2)
        if FADING:
            channel = Rayleigh(awgn_std=std)
        else:
            channel = AWGN(standard_dev=std)

        rx_images = np.empty(images[:number_sims].shape, dtype=np.float32)
        bit_errors = 0

        for k, image in enumerate(images[sim_idx]):

            # reshape the image to be a 1D array
            floats = image.flatten()
            
            # convert the message to bits (not NaN values)
            data_bits = np.empty((quant_bits*3*IM_SIZE**2,), dtype=np.uint8)
            for i, value in enumerate(floats):
                val_bits = qnt.quantize(value)
                data_bits[i*quant_bits:(i+1)*quant_bits] = val_bits

            # encode, modulate, transmit, receive, demodulate, decode
            if channel_coding:
                tx_bits = phy.conv_encode(data_bits)
            else:
                tx_bits = data_bits
            tx_symbols = phy.qam_modulate(tx_bits)
            if FADING:
                fade_symbols, fade_coefs = channel.fade_signal(tx_symbols)
            else:
                fade_symbols = tx_symbols
            if modulation_type == 'BPSK':
                rx_symbols = channel.add_real_noise(fade_symbols)
            else:
                rx_symbols = channel.add_complex_noise(fade_symbols)
            if FADING:
                eq_symbols = np.divide(rx_symbols, fade_coefs)
            else:
                eq_symbols = rx_symbols
            if channel_coding and vitdec_type == 'soft':
                rx_data_bits = phy.viterbi_decode(eq_symbols, dec_type='soft')
            elif channel_coding and vitdec_type == 'hard':
                rx_bits = phy.qam_demodulate(eq_symbols)
                rx_data_bits = phy.viterbi_decode(rx_bits, dec_type='hard')
            else:
                rx_data_bits = phy.qam_demodulate(eq_symbols)

            # convert the received bits to floats (filling in NaN values)
            rx_floats = np.empty((3*IM_SIZE**2,), dtype=np.float32)
            j = 0
            for i in range(3*IM_SIZE**2):
                try:
                    rx_floats[i] = qnt.dequantize(
                        rx_data_bits[i*quant_bits:(i+1)*quant_bits]
                    )
                except ValueError:
                    breakpoint()

            # reshape the received floats back to an image
            rx_image = rx_floats.reshape((IM_SIZE, IM_SIZE, N_CHN))

            # save off the received sample for later decoding
            rx_images[k] = rx_image

            # accumulate the results
            bit_errors += np.sum(np.abs(data_bits - rx_data_bits))

            print(f'\rSample {k+1}/{number_sims} processed.', end='')

        # clip the received images to [-1, 1]
        rx_images = np.clip(rx_images, -1, 1)

        # decode the received samples
        print('\nDecoding...')
        preds = classifier.predict(rx_images, verbose=0)

        print(f'Finished EBN0 = {EBN0} dB.\n')

        preds = np.argmax(preds, axis=1)
        true = np.argmax(labels[sim_idx], axis=1)

        bers[m] = bit_errors / (number_sims*3*32*IM_SIZE**2)
        accs[m] = np.mean(preds == true)

    return accs, bers

# ------------------------------------------------------------------------------

def simulate_effective(number_sims: int = 500,
                       snrs: np.ndarray = np.linspace(-20, 20, 7),
                       task: str = 'shapes', modulation_type: str = 'BPSK',
                       channel_coding: bool = False, code_rate: str = '1/2',
                       vitdec_type: str = 'soft', code_bits: int = 32,
                       channel: str = 'awgn'):
    """
    This function simulates the effective system.

    Parameters
    ----------
    number_sims : int
        The number of simulated messages to run for each trial.
        Default is 500.
    snrs : np.ndarray
        The SNR values to simulate.
        Default is np.linspace(-20, 20, 7).
    task : str
        The task to simulate.
        Default is 'shapes'.
    modulation_type : str
        The modulation type to use.
        Default is 'BPSK'.
    channel_coding : bool
        Whether to use channel coding or not.
        Default is False.
    code_rate : str
        The code rate to use. Only used if channel_coding is True.
        Default is '1/2'.
    vitdec_type : str
        The type of Viterbi decoder to use. Only used if channel_coding is True.
        Default is 'soft'.
    code_bits : int
        The number of bits to use for the code.
        Default is 32.
    channel : str
        The channel to use. Options are 'awgn' and 'rayleigh'.
        Default is 'awgn'.
    """

    # --- constants ---
    
    N_IMS = 2630
    IM_SIZE = 64
    N_CHN = 3
    if task in ['shapes', 'colors']:
        N_CLS = 4
    elif task == 'isSpeedLimit':
        N_CLS = 2

    model_path = f'local/models/classifiers/{task}.keras'
    images_path = 'local/memmap_data/signs_tst_data.npy'
    labels_path = f'local/memmap_data/signs_tst_labels_{task}.npy'

    EB          = 1.0

    if channel == 'awgn':
        FADING = False
    elif channel == 'rayleigh':
        FADING = True
    else:
        raise ValueError(f'Invalid channel! Got {channel}')

    # --- load data ---

    images = np.memmap(images_path, dtype=np.float32, mode='r', 
                       shape=(N_IMS, IM_SIZE, IM_SIZE, N_CHN))

    labels = np.memmap(labels_path, dtype=np.float32, mode='r', 
                       shape=(N_IMS, N_CLS))

    # --- instantiate components ---

    phy = PHY(modulation_type, code_rate)
    classifier = keras.models.load_model(model_path)
    block_coder = M2LinearBlockCode(code_bits)

    # --- simulate the system ---

    bers = np.empty(snrs.shape, dtype=np.float32)
    accs = np.empty(snrs.shape, dtype=np.float32)

    sim_idx = np.random.choice(N_IMS, number_sims, replace=False)

    # --- get the predicted labels ---

    preds = classifier.predict(images[sim_idx], verbose=0)
    int_preds = np.argmax(preds, axis=1)

    for m, EBN0 in enumerate(snrs):

        N0 = EB / 10**(EBN0/10)
        std = np.sqrt(N0/2)
        if FADING:
            channel = Rayleigh(awgn_std=std)
        else:
            channel = AWGN(standard_dev=std)

        rx_int_preds = np.empty(number_sims, dtype=np.uint32)
        bit_errors = 0

        for k, pred in enumerate(int_preds):
            
            data_bits = block_coder.encode(pred)

            # encode, modulate, transmit, receive, demodulate, decode
            if channel_coding:
                tx_bits = phy.conv_encode(data_bits)
            else:
                tx_bits = data_bits
            tx_symbols = phy.qam_modulate(tx_bits)
            if FADING:
                fade_symbols, fade_coefs = channel.fade_signal(tx_symbols)
            else:
                fade_symbols = tx_symbols
            if modulation_type == 'BPSK':
                rx_symbols = channel.add_real_noise(fade_symbols)
            else:
                rx_symbols = channel.add_complex_noise(fade_symbols)
            if FADING:
                eq_symbols = np.divide(rx_symbols, fade_coefs)
            else:
                eq_symbols = rx_symbols
            if channel_coding and vitdec_type == 'soft':
                rx_data_bits = phy.viterbi_decode(eq_symbols, dec_type='soft')
            elif channel_coding and vitdec_type == 'hard':
                rx_bits = phy.qam_demodulate(eq_symbols)
                rx_data_bits = phy.viterbi_decode(rx_bits, dec_type='hard')
            else:
                rx_data_bits = phy.qam_demodulate(eq_symbols)

            rx_int_preds[k] = block_coder.decode(rx_data_bits)

            # accumulate the results
            bit_errors += np.sum(np.abs(data_bits - rx_data_bits))

            print(f'\rSample {k+1}/{number_sims} processed.', end='')

        print(f'Finished EBN0 = {EBN0} dB.\n')

        true = np.argmax(labels[sim_idx], axis=1)

        bers[m] = bit_errors / (number_sims*code_bits)
        accs[m] = np.mean(rx_int_preds == true)

    return accs, bers

# ------------------------------------------------------------------------------

def simulate_end_to_end(number_sims: int = 500, 
                        snrs: np.ndarray = np.linspace(-20, 20, 7), 
                        task: str = 'shapes', message_length: int = 2,
                        modulation_type: str = 'BPSK', channel: str = 'awgn'):
    """
    This function simulates the end-to-end system.

    Parameters
    ----------
    number_sims : int
        The number of simulated messages to run for each trial.
        Default is 500.
    snrs : np.ndarray
        The SNR values to simulate.
        Default is np.linspace(-20, 20, 7).
    task : str
        The task to simulate.
        Default is 'shapes'.
    message_length : int
        The length of the message to send.
        Default is 2.
    modulation_type : str
        The modulation type to use.
        Default is 'BPSK'.
    channel : str
        The channel to use. Options are 'awgn' and 'rayleigh'.
        Default is 'awgn'.
    """
    
    # --- constants ---
    
    N_IMS = 2630
    IM_SIZE = 64
    N_CHN = 3
    
    if task in ['shapes', 'colors']:
        N_CLS = 4
    elif task == 'isSpeedLimit':
        N_CLS = 2

    if channel == 'awgn':
        FADING = False
    elif channel == 'rayleigh':
        FADING = True
    else:
        raise ValueError(f'Invalid channel! Got {channel}')

    tsk = task
    m_len = message_length
    mod = modulation_type

    encoder_path = f'local/models/end_to_end/len{m_len}_{mod}_{channel}_encoder.keras'
    decoder_path = f'local/models/end_to_end/{tsk}_len{m_len}_{mod}_{channel}_decoder.keras'
    images_path = 'local/memmap_data/signs_tst_data.npy'
    labels_path = f'local/memmap_data/signs_tst_labels_{tsk}.npy'

    EB          = 1.0

    # --- load data ---

    images = np.memmap(images_path, dtype=np.float32, mode='r', 
                       shape=(N_IMS, IM_SIZE, IM_SIZE, N_CHN))

    labels = np.memmap(labels_path, dtype=np.float32, mode='r', 
                       shape=(N_IMS, N_CLS))

    # --- instantiate components ---

    phy = PHY(modulation_type)
    encoder = keras.models.load_model(encoder_path)
    decoder = keras.models.load_model(decoder_path)

    # --- simulate the system ---

    accs = np.empty(snrs.shape, dtype=np.float32)

    sim_idx = np.random.choice(N_IMS, number_sims, replace=False)

    # --- get the coded bits ---

    coded_bits = encoder.predict(images[sim_idx], verbose=0)
    coded_bits = coded_bits/2.0 + 0.5

    for m, EBN0 in enumerate(snrs):

        N0 = EB / 10**(EBN0/10)
        std = np.sqrt(N0/2)
        if FADING:
            channel = Rayleigh(awgn_std=std)
        else:
            channel = AWGN(standard_dev=std)

        if modulation_type == 'BPSK':
            rx_dim = coded_bits.shape[1]
        elif modulation_type == '16-QAM':
            rx_dim = coded_bits.shape[1] // 2
        else:
            raise ValueError('Invalid modulation type.')

        all_rx_symbols = np.empty((number_sims,rx_dim), dtype=np.float32)

        for k, tx_bits in enumerate(coded_bits):
            
            tx_symbols = phy.qam_modulate(tx_bits)
            if FADING:
                fade_symbols, fade_coefs = channel.fade_signal(tx_symbols)
            else:
                fade_symbols = tx_symbols
            if modulation_type == 'BPSK':
                rx_symbols = channel.add_real_noise(fade_symbols)
                if FADING:
                    eq_symbols = np.divide(rx_symbols, fade_coefs)
                else:
                    eq_symbols = rx_symbols
            else:
                rx_symbols = channel.add_complex_noise(fade_symbols)
                if FADING:
                    eq_symbols = np.divide(rx_symbols, fade_coefs)
                else:
                    eq_symbols = rx_symbols
                eq_symbols = np.concatenate((eq_symbols.real, eq_symbols.imag))
            all_rx_symbols[k] = eq_symbols

            print(f'\rSample {k+1}/{number_sims} processed.', end='')

        print('\nDecoding...')
        decoded = decoder.predict(all_rx_symbols, verbose=0)

        print(f'Finished EBN0 = {EBN0} dB.\n')

        preds = np.argmax(decoded, axis=1)
        true = np.argmax(labels[sim_idx], axis=1)

        accs[m] = np.mean(preds == true)

    return accs

# ------------------------------------------------------------------------------

if __name__=="__main__":
    
    effect_matrix = matrices.sc_shapes_modpc
    n_sims = 25
    message_length = 2

    SNRS = np.linspace(-40, 20, 15)

    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()

    # --- sccs-r ---

    snrs, bers, accs = simulate_sccsr(
        effect_matrix, 
        number_sims=n_sims,
        message_length=message_length,
        snrs=SNRS
    )

    ax1.plot(snrs, accs, marker='o', linestyle='--', color='blue', 
             label='SCCS-R')
    
    ax2 = ax1.twinx()
    ax2.plot(snrs, bers, marker='s', linestyle=':', color='red', label='BER')
    ax2.set_ylabel('BER')
    ax2.set_yscale('log')  # Set logarithmic scale for BER
    ax2.legend(loc='center right')

    # --- technical ---

    keras.backend.clear_session()

    snrs, bers1, accs = simulate_technical(number_sims=n_sims, snrs=SNRS)

    ax1.plot(snrs, accs, marker='o', linestyle='--', color='green', 
             label='Technical')
    
    # --- effective ---
    
    keras.backend.clear_session()

    snrs, bers2, accs = simulate_effective(
        number_sims=n_sims, snrs=SNRS, quant_bits=2)

    ax1.plot(snrs, accs, marker='o', linestyle='--', color='purple',
                label='Effective')
    
    # --- end-to-end ---
    
    keras.backend.clear_session()
    
    snrs, accs = simulate_end_to_end(
        number_sims=n_sims,
        snrs=SNRS,
        task='shapes',
        modulation_type='BPSK'
    )

    ax1.plot(snrs, accs, marker='o', linestyle='--', color='orange',
                label='End-to-End')
    
    ax1.set_xlabel('Eb/N0 (dB)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Performance Metrics')
    ax1.grid(True)
    ax1.legend(loc='center left')

    plt.show()

    # --- plot the bers on another graph ---

    plt.figure(figsize=(8, 6))
    plt.semilogy(snrs, bers1, marker='s', linestyle=':', color='red', label='Technical')
    plt.semilogy(snrs, bers2, marker='s', linestyle=':', color='purple', label='Effective')
    plt.semilogy(snrs, bers, marker='s', linestyle=':', color='orange', label='SCCS-r')

    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Performance Metrics')
    plt.grid(True)
    plt.legend(loc='center right')
    plt.show()

# ------------------------------------------------------------------------------