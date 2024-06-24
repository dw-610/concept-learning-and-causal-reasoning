"""
This script simulates the full SCCS-R system with trained encoder, decoder,
reasoner and channel models to evaluate the effect of the message length in the
system.
"""

# ------------------------------------------------------------------------------
# import

print("\nSilencing Tensorflow info, warnings, and errors...\n")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from modules.comms.wifi_phy import PHY
from modules.comms.channels import AWGN
from modules.comms.quantizer import Quantizer

from modules.causal_cs.reasoning import Reasoner
from modules.causal_cs.matrices import mats as matrices

# ------------------------------------------------------------------------------

def main(task: str, reasoning: bool, message_length: int):
    
    if task in ['shapes', 'colors']:
        EFFECT_MAT  = matrices[f'sc_{task}']
    elif task == 'isSpeedLimit':
        EFFECT_MAT  = matrices[f'scy_{task}']
    else:
        raise ValueError(f'Invalid task! Got {task}')

    DATA_DIR    = 'local/causal_data/'
    if task in ['shapes', 'colors']:
        DATA_FILE   = f'tst_shapes-colors_{task}.csv'
    elif task == 'isSpeedLimit':
        DATA_FILE   = f'tst_shapes-colors-symbols_{task}.csv'
    else:
        raise ValueError(f'Invalid task! Got {task}')

    MODELS_DIR  = 'local/models/decoders/'
    MODEL_NAME  = f'{task}.keras'

    MOD_TYPE    = 'BPSK'
    CH_CODING   = False
    CODE_RATE   = '1/2'
    VITDEC_TYPE = 'soft'
    EB          = 1.0
    EBN0        = 5.0
    Q_BITS      = 8
    Q_RANGE     = (-5, 5)

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

    N0 = EB / 10**(EBN0/10)
    std = np.sqrt(N0/2)
    channel = AWGN(standard_dev=std)

    phy = PHY(MOD_TYPE, CODE_RATE)
    qnt = Quantizer(Q_BITS, Q_RANGE)
    reasoner = Reasoner(EFFECT_MAT)
    decoder = tf.keras.models.load_model(MODELS_DIR + MODEL_NAME)

    # --- simulate the system ---

    rx_samples = np.empty(data.shape, dtype=np.float32)

    for k, sample in enumerate(data):

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
        if CH_CODING:
            tx_bits = phy.conv_encode(data_bits)
        else:
            tx_bits = data_bits
        tx_symbols = phy.qam_modulate(tx_bits)
        if MOD_TYPE == 'BPSK':
            rx_symbols = channel.add_real_noise(tx_symbols)
        else:
            rx_symbols = channel.add_complex_noise(tx_symbols)
        if CH_CODING and VITDEC_TYPE == 'soft':
            rx_data_bits = phy.viterbi_decode(rx_symbols, dec_type='soft')
        elif CH_CODING and VITDEC_TYPE == 'hard':
            rx_bits = phy.qam_demodulate(rx_symbols)
            rx_data_bits = phy.viterbi_decode(rx_bits, dec_type='hard')
        else:
            rx_data_bits = phy.qam_demodulate(rx_symbols)

        # convert the received bits to floats (filling in NaN values)
        rx_sample = np.empty((n_dims,), dtype=np.float32)
        j = 0
        for i in range(n_dims):
            if i in nan_idx:
                rx_sample[i] = np.nan
            else:
                rx_sample[i] = qnt.dequantize(
                    rx_data_bits[j*Q_BITS:(j+1)*Q_BITS]
                )
                j += 1

        # save off the received sample for later decoding
        rx_samples[k] = rx_sample

        print(f'\rSample {k+1}/{n_samples} processed.', end='')

    # decode the received samples
    print('\nDecoding...')
    decoded = decoder.predict(rx_samples, verbose=0)

    if labels.shape[1] > 1:
        true = np.argmax(labels, axis=1)
        preds = np.argmax(decoded, axis=1)
    else:
        true = labels.flatten()
        preds = np.round(decoded.flatten())

    accuracy = np.mean(preds == true)

    return accuracy

    # --- end ---
    



# ------------------------------------------------------------------------------

if __name__ == '__main__':

    SAVE_DIR = 'local/figures/simulations/'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # simulation params
    TASKS       = ['shapes', 'colors', 'isSpeedLimit']
    SC_LENGTHS  = [int(i) for i in range(1, 6)]
    SL_LENGTHS  = [int(i) for i in range(1, 21)]

    # plotting params
    FIG_SIZE        = (3.5, 2.5)
    DPI             = 300
    MARKER_SIZE     = 5
    LINE_WIDTH      = 1
    R_LINESTYLE     = '--'
    NO_R_LINESTYLE  = ':'
    R_MARKER        = 'o'
    NO_R_MARKER     = '^'
    R_COLOR         = 'blue'
    NO_R_COLOR      = 'red'
    FONT_SIZE       = 10
    LABEL_SIZE      = 8

    for task in TASKS:
        print(f'\n ===== Task = {task} ===== \n')

        plt.figure(figsize=FIG_SIZE, dpi=DPI)

        if task in ['shapes', 'colors']:
            lengths = SC_LENGTHS
            ticks = lengths
        elif task == 'isSpeedLimit':
            lengths = SL_LENGTHS
            ticks = [i for i in range(0, 21, 2)]
        else:
            raise ValueError(f'Invalid task! Got {task}')

        wr_ACCs = np.empty((len(lengths),), dtype=np.float32)
        wor_ACCs = np.empty((len(lengths),), dtype=np.float32)

        for i, length in enumerate(lengths):
            print(f'\n --- Message length = {length} --- \n')

            wr_ACCs[i] = main(task, reasoning=True, message_length=length)
            wor_ACCs[i] = main(task, reasoning=False, message_length=length)

        plt.plot(lengths, wr_ACCs, 
                marker=R_MARKER, linestyle=R_LINESTYLE, color=R_COLOR, 
                label='Reasoning', markerfacecolor='none',linewidth=LINE_WIDTH, 
                     markersize=MARKER_SIZE)

        plt.plot(lengths, wor_ACCs, 
                marker=NO_R_MARKER, linestyle=NO_R_LINESTYLE, color=NO_R_COLOR, 
                label='Random',markerfacecolor='none',linewidth=LINE_WIDTH, 
                     markersize=MARKER_SIZE)

        plt.xlabel('Message Length', fontsize=FONT_SIZE)
        plt.ylabel('Accuracy', fontsize=FONT_SIZE)
        plt.xticks(ticks)
        plt.grid()
        plt.legend(fontsize='small', loc='best')
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)

        plt.savefig(SAVE_DIR + f'/{task}_length_effect.pdf')

# ------------------------------------------------------------------------------