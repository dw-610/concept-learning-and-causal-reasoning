"""
This script simulates the full SCCS-R system with trained encoder, decoder,
reasoner and channel models to evaluate the effect of including reasoning in the
system, vs dropping random semantic representation values.
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
from matplotlib.lines import Line2D

from modules.comms.wifi_phy import PHY
from modules.comms.channels import AWGN
from modules.comms.quantizer import Quantizer

from modules.causal_cs.reasoning import Reasoner
from modules.causal_cs.matrices import mats as matrices

# ------------------------------------------------------------------------------

def main(task: str, reasoning: bool, message_length: int):

    if task == 'shapes':
        EFFECT_MAT  = matrices['sc_shapes']
    elif task == 'colors':
        EFFECT_MAT  = matrices['sc_colors']
    elif task == 'isSpeedLimit':
        EFFECT_MAT  = matrices['scy_isSpeedLimit']
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

    phy = PHY(MOD_TYPE, CODE_RATE)
    qnt = Quantizer(Q_BITS, Q_RANGE)
    reasoner = Reasoner(EFFECT_MAT)
    decoder = tf.keras.models.load_model(MODELS_DIR + MODEL_NAME)

    # --- simulate the system ---

    EBN0s = np.linspace(-40, 20, 7)

    BERs = np.empty(EBN0s.shape, dtype=np.float32)
    ACCs = np.empty(EBN0s.shape, dtype=np.float32)

    for m, EBN0 in enumerate(EBN0s):

        N0 = EB / 10**(EBN0/10)
        std = np.sqrt(N0/2)
        channel = AWGN(standard_dev=std)

        rx_samples = np.empty(data.shape, dtype=np.float32)
        bit_errors = 0

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

            # accumulate the results
            bit_errors += np.sum(np.abs(data_bits - rx_data_bits))

            print(f'\rSample {k+1}/{n_samples} processed.', end='')

        # decode the received samples
        print('\nDecoding...')
        decoded = decoder.predict(rx_samples, verbose=0)

        print(f'Finished EBN0 = {EBN0} dB.\n')

        if labels.shape[1] > 1:
            true = np.argmax(labels, axis=1)
            preds = np.argmax(decoded, axis=1)
        else:
            true = labels.flatten()
            preds = np.round(decoded.flatten())

        BERs[m] = bit_errors / (n_samples * n_dims * 32)
        ACCs[m] = np.mean(preds == true)

    return EBN0s, BERs, ACCs

    # --- end ---
    



# ------------------------------------------------------------------------------

if __name__ == '__main__':

    SAVE_DIR = 'local/figures/simulations/'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # simulation params
    LENGTHS     = [int(i) for i in range(1, 5)]
    TASKS       = ['shapes', 'colors']

    # plotting params
    FIG_SIZE        = (3.5, 2.5)
    DPI             = 300
    MARKER_SIZE     = 5
    LINE_WIDTH      = 1
    R_LINESTYLE     = '--'
    NO_R_LINESTYLE  = ':'
    MARKERS         = ['o', '^', 's', 'D']
    FONT_SIZE       = 10
    LABEL_SIZE      = 8

    for task in TASKS:
        print(f'\n ===== Task = {task} ===== \n')

        plt.figure(figsize=FIG_SIZE, dpi=DPI)

        for i, length in enumerate(LENGTHS):
            print(f'\n --- Message length = {length} --- \n')

            EBN0s, _, ACCs = main(task=task, reasoning=True, 
                                  message_length=length)

            plt.plot(EBN0s, ACCs, 
                    marker=MARKERS[i], linestyle=R_LINESTYLE, color='blue', 
                    markerfacecolor='none', linewidth=LINE_WIDTH, 
                    markersize=MARKER_SIZE)
            
            EBN0s, _, ACCs = main(task=task, reasoning=False, 
                                  message_length=length)
            
            plt.plot(EBN0s, ACCs, marker=MARKERS[i], linestyle=NO_R_LINESTYLE, 
                     color='red', markerfacecolor='none',linewidth=LINE_WIDTH, 
                     markersize=MARKER_SIZE)
            
        plt.xlabel('$E_b/N_0$ (dB)', fontsize=FONT_SIZE)
        plt.ylabel('Accuracy', fontsize=FONT_SIZE)
        plt.grid()
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
 
        # custom legend
        custom_lines = [Line2D([0], [0], color='black', linestyle='', marker=mk, 
                        markerfacecolor='none', 
                        markersize=MARKER_SIZE) for mk in MARKERS]
        custom_lines.append(Line2D([0], [0], color='blue', linestyle='', 
                                   marker='s', markersize=MARKER_SIZE))
        custom_lines.append(Line2D([0], [0], color='red', linestyle='', 
                                   marker='s', markersize=MARKER_SIZE))
        plt.legend(custom_lines, 
                   [f'Length {i}' for i in range(1,5)]+['Reasoning', 'Random'], 
                   fontsize='small', loc='upper left', ncol=1)
        
        plt.savefig(SAVE_DIR + f'/{task}_reasoning_efficacy.pdf')

    TASKS = ['isSpeedLimit']
    LENGTHS = list(range(7, 20, 4))

    for task in TASKS:
        print(f'\n ===== Task = {task} ===== \n')

        plt.figure(figsize=FIG_SIZE, dpi=DPI)

        for i, length in enumerate(LENGTHS):
            print(f'\n --- Message length = {length} --- \n')

            EBN0s, _, ACCs = main(task=task, reasoning=True, 
                                  message_length=length)

            plt.plot(EBN0s, ACCs, 
                    marker=MARKERS[i], linestyle=R_LINESTYLE, color='blue', 
                    markerfacecolor='none', linewidth=LINE_WIDTH, 
                    markersize=MARKER_SIZE)
            
            EBN0s, _, ACCs = main(task=task, reasoning=False, 
                                  message_length=length)
            
            plt.plot(EBN0s, ACCs, marker=MARKERS[i], linestyle=NO_R_LINESTYLE, 
                     color='red', markerfacecolor='none',linewidth=LINE_WIDTH, 
                     markersize=MARKER_SIZE)
            
        plt.xlabel('$E_b/N_0$ (dB)', fontsize=FONT_SIZE)
        plt.ylabel('Accuracy', fontsize=FONT_SIZE)
        plt.grid()
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
 
        # custom legend
        custom_lines = [Line2D([0], [0], color='black', linestyle='', marker=mk, 
                        markerfacecolor='none', 
                        markersize=MARKER_SIZE) for mk in MARKERS]
        custom_lines.append(Line2D([0], [0], color='blue', linestyle='', 
                                   marker='s', markersize=MARKER_SIZE))
        custom_lines.append(Line2D([0], [0], color='red', linestyle='', 
                                   marker='s', markersize=MARKER_SIZE))
        plt.legend(custom_lines, 
                   [f'Length {i}' for i in range(7,20,4)]+['Reasoning', 'Random'], 
                   fontsize='small', loc='upper left', ncol=1)
        
        plt.savefig(SAVE_DIR + f'/{task}_reasoning_efficacy.pdf')

# ------------------------------------------------------------------------------