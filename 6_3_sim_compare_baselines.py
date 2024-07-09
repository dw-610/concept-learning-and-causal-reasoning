"""
This script will simulate the SCCS-R and baseline systems and compare their
performance.
"""

# ------------------------------------------------------------------------------
# imports

import os
import keras
import numpy as np
import matplotlib.pyplot as plt

from modules.causal_cs import matrices
import modules.simulation as sim

# ------------------------------------------------------------------------------

def main(task: str, n_sims: int, message_length: int, channel: str,
         mod_type: str):

    SAVE_DIR    = 'local/figures/simulations/'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # simulation params
    WHICH       = ['e2e', 'sccsr', 'eff', 'tech']
    CH_CODING   = False
    CODE_RATE   = '1/2'
    VITDEC_TYPE = 'soft'
    REASONING   = True
    Q_BITS      = 8
    Q_RANGE     = (-5, 5)
    EFF_BITS    = Q_BITS * message_length

    SNRS        = np.linspace(-80, 40, 11)

    if task in ['shapes', 'colors']:
        EFF_MATRIX  = matrices.mats[f'sc_{task}']
    elif task == 'isSpeedLimit':
        EFF_MATRIX  = matrices.mats['scy_isSpeedLimit']
    else:
        raise ValueError(f'Invalid task! Got {task}')
    
    # plotting params
    FIG_SIZE        = (3.5, 2.5)
    DPI             = 300
    MARKER_SIZE     = 5
    LINE_WIDTH      = 1
    MARKERS         = ['x', 's', '^', 'o']
    COLORS          = ['green', 'red', 'purple', 'blue']
    LINE_STYLE      = '--'
    FONT_SIZE       = 10
    LABEL_SIZE      = 8

    # --- initialize the plot ---

    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax1 = plt.gca()

    # --- do the simulations ---

    if 'sccsr' in WHICH:
        accs_sccsr, _ = sim.simulate_sccsr(
            effect_matrix=EFF_MATRIX, number_sims=n_sims, snrs=SNRS, 
            reasoning=REASONING, message_length=message_length, task=task,
            modulation_type=mod_type, channel_coding=CH_CODING, 
            code_rate=CODE_RATE, vitdec_type=VITDEC_TYPE, quant_bits=Q_BITS, 
            quant_range=Q_RANGE, channel=channel)
        keras.backend.clear_session()

    if 'tech' in WHICH:
        accs_tech, _ = sim.simulate_technical(
            number_sims=n_sims, snrs=SNRS, task=task, modulation_type=mod_type,
            channel_coding=CH_CODING, code_rate=CODE_RATE, 
            vitdec_type=VITDEC_TYPE, quant_bits=Q_BITS, quant_range=Q_RANGE,
            channel=channel)
        keras.backend.clear_session()

    if 'eff' in WHICH:
        accs_eff, _ = sim.simulate_effective(
            number_sims=n_sims, snrs=SNRS, task=task, modulation_type=mod_type,
            channel_coding=CH_CODING, code_rate=CODE_RATE, 
            vitdec_type=VITDEC_TYPE, code_bits=EFF_BITS, channel=channel)
        keras.backend.clear_session()
    
    if 'e2e' in WHICH:
        accs_e2e = sim.simulate_end_to_end(
            number_sims=n_sims, snrs=SNRS, task=task, 
            message_length=message_length, modulation_type=mod_type, 
            channel=channel)

    # --- plot the results ---

    if 'sccsr' in WHICH:
        ax1.plot(SNRS, accs_sccsr, marker=MARKERS[0], linestyle=LINE_STYLE, color=COLORS[0], 
                label='SCCS-R', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, markerfacecolor='none')
    if 'tech' in WHICH:
        ax1.plot(SNRS, accs_tech, marker=MARKERS[1], linestyle=LINE_STYLE, color=COLORS[1], 
                label='Technical', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, markerfacecolor='none')
    if 'eff' in WHICH:
        ax1.plot(SNRS, accs_eff, marker=MARKERS[2], linestyle=LINE_STYLE, color=COLORS[2],
                    label='Effective', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, markerfacecolor='none')
    if 'e2e' in WHICH:
        ax1.plot(SNRS, accs_e2e, marker=MARKERS[3], linestyle=LINE_STYLE, color=COLORS[3],
                    label='End-to-End', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, markerfacecolor='none')
    
    ax1.set_xlabel('$E_B/N_0$ (dB)', fontsize=FONT_SIZE)
    ax1.set_ylabel('Accuracy', fontsize=FONT_SIZE)
    ax1.grid(True)
    ax1.set_ylim((0, 1))
    ax1.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)

    plt.savefig(f'local/figures/simulations/{task}_compare_all_{channel}_{mod_type}.pdf')

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    N_SIMS = 1000

    TASKS = ['shapes', 'colors']
    CHANNELS = ['awgn']
    MOD_TYPE = ['BPSK']
    MESSAGE_LENGTH = 2

    for task in TASKS:
        for channel, mod_type in zip(CHANNELS, MOD_TYPE):
            main(task, n_sims=N_SIMS, message_length=MESSAGE_LENGTH, 
                 channel=channel, mod_type=mod_type)
    
    TASKS = ['isSpeedLimit']
    CHANNELS = ['awgn']
    MOD_TYPE = ['BPSK']
    MESSAGE_LENGTH = 10

    for task in TASKS:
        for channel, mod_type in zip(CHANNELS, MOD_TYPE):
            main(task, n_sims=N_SIMS, message_length=MESSAGE_LENGTH, 
                 channel=channel, mod_type=mod_type)
            
    TASKS = ['shapes']
    CHANNELS = ['rayleigh']
    MOD_TYPE = ['16-QAM']
    MESSAGE_LENGTH = 2

    for task in TASKS:
        for channel, mod_type in zip(CHANNELS, MOD_TYPE):
            main(task, n_sims=N_SIMS, message_length=MESSAGE_LENGTH, 
                 channel=channel, mod_type=mod_type)

# ------------------------------------------------------------------------------