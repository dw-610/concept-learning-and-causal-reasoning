"""
This script implements the partial SCCS-R system to get data for the decoder.

The system is as follows:
    - Semantic encoder
    - Reasoner
    - Channel

The semantic encoder has already been implemented.

The reasoner is a simple function defined in causal_cs.reasoning.py. However,
for the purpose of generating data, the reasoner will not be used. Instead,
random values will be set to NaN.

The channel can be AWGN, Rayleigh or Rician, with SNR randomly chosen for each
data sample in the range of -20 to 20 dB.
"""

# ------------------------------------------------------------------------------
# imports

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modules.comms.wifi_phy import PHY
from modules.comms.channels import AWGN
from modules.comms.quantizer import Quantizer

# ------------------------------------------------------------------------------

def main(task: str, multiplier: int = 10):

    EB          = 1

    MOD_TYPE    = 'BPSK'
    CH_CODING   = False
    CODE_RATE   = '1/2'
    VITDEC_TYPE = 'soft'
    Q_BITS      = 8
    Q_RANGE     = (-5, 5)

    EBN0_RANGE  = (-20, 20)
    
    DATA_DIR    = 'local/'
    LOAD_SUBDIR = 'causal_data/'
    SAVE_SUBDIR = 'decoder_data/'
    if task in ['shapes', 'colors']:
        DATA_FILE   = f'trn_shapes-colors_{task}.csv'
    elif task == 'isSpeedLimit':
        DATA_FILE   = f'trn_shapes-colors-symbols_{task}.csv'
    else:
        raise ValueError(f'Invalid task! Got {task}')
    
    if not os.path.exists(DATA_DIR + SAVE_SUBDIR):
        os.makedirs(DATA_DIR + SAVE_SUBDIR)

    # --- load and format data ---

    df = pd.read_csv(DATA_DIR+ LOAD_SUBDIR + DATA_FILE)

    if task in ['shapes', 'colors']:
        data = df[['s0', 's1', 'c0', 'c1', 'c2']].to_numpy().astype(np.float32)
        labels = df[['u0', 'u1', 'u2', 'u3']].to_numpy().astype(np.float32)
    elif task == 'isSpeedLimit':
        data = df[['s0', 's1', 'c0', 'c1', 'c2', 'y0', 'y1', 'y2', 'y3', 'y4',
                    'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13', 
                    'y14']].to_numpy().astype(np.float32)
        labels = df[['u']].to_numpy().astype(np.float32)

    n_samples, n_dims = data.shape

    # --- instantiate communication system components ---

    phy = PHY(MOD_TYPE, CODE_RATE)
    ch = AWGN()
    qnt = Quantizer(Q_BITS, Q_RANGE)

    # --- generate data ---

    new_data = np.zeros((n_samples*multiplier, n_dims), dtype=np.float32)
    new_labels = np.tile(labels, (multiplier, 1))

    k = 0
    for _ in range(multiplier):
        for sample in data:

            sample = sample.copy()

            # randomly set some values to NaN
            if task in ['shapes', 'colors']:
                msg_len = np.random.randint(1, n_dims+1)
            elif task == 'isSpeedLimit':
                msg_len = np.random.randint(10, n_dims+1)
            nan_idx = np.random.choice(n_dims, n_dims-msg_len, replace=False)
            sample[nan_idx] = np.nan

            # randomly set the SNR
            EbN0 = np.random.uniform(EBN0_RANGE[0], EBN0_RANGE[1])
            N0 = EB / 10**(EbN0/10)
            ch.std = np.sqrt(N0/2)
            
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
                rx_symbols = ch.add_real_noise(tx_symbols)
            else:
                rx_symbols = ch.add_complex_noise(tx_symbols)
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
                        rx_data_bits[j*Q_BITS:(j+1)*Q_BITS])
                    j += 1

            new_data[k] = rx_sample
            k += 1

            if k % 500 == 0:
                print(f'\rProcessed {k} samples.', end='')
    print()

    # --- split data into training and validation sets ---

    trn_data, vld_data, trn_labels, vld_labels = \
        train_test_split(new_data, new_labels, test_size=0.2)
    
    # --- save data ---

    np.save(DATA_DIR + SAVE_SUBDIR + f'{task}_trn_data.npy', trn_data)
    np.save(DATA_DIR + SAVE_SUBDIR + f'{task}_vld_data.npy', vld_data)
    np.save(DATA_DIR + SAVE_SUBDIR + f'{task}_trn_labels.npy', trn_labels)
    np.save(DATA_DIR + SAVE_SUBDIR + f'{task}_vld_labels.npy', vld_labels)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    TASKS = ['shapes', 'colors', 'isSpeedLimit']
    MULTIPLIER = 10

    for task in TASKS:
        print(f'Generating data for task: {task}')
        main(task, MULTIPLIER)

# ------------------------------------------------------------------------------