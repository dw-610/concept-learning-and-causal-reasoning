"""
This script uses the DirectLiNGAM algorithm from the lingam package to estimate
the causal graph for the specific datasets of the German traffic sign data.
"""

# ------------------------------------------------------------------------------
# imports

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from lingam import DirectLiNGAM

# ------------------------------------------------------------------------------

def main(which_label: str, save_plots: bool = False, show_plots: bool = False, 
         verbose: bool = False):

    MEASURE     = 'pwling'   
    SEED        = 610

    SAVE_DIR = 'local/figures/causal_discovery/'
    SAVE_FILE = f'{which_label}-{MEASURE}'

    SHAPE_DIM   = 2
    COLOR_DIM   = 3
    SYMBOL_DIM  = 15

    if which_label in ['shapes', 'colors']:
        NUM_LABELS = 4
    elif which_label == 'isSpeedLimit':
        NUM_LABELS = 1
    else:
        raise ValueError(f'Invalid label! Got {which_label}')

    DATA_DIR    = 'local/causal_data/'
    if which_label in ['shapes', 'colors']:
        DATA_FILE = f'trn_shapes-colors_{which_label}.csv'
        NODE_COLORS = ['blue']*SHAPE_DIM + ['green']*COLOR_DIM + \
            ['red']*NUM_LABELS
        NUM_DATA    = SHAPE_DIM + COLOR_DIM
        NUM_TOTAL   = SHAPE_DIM + COLOR_DIM + NUM_LABELS
    elif which_label == 'isSpeedLimit':
        DATA_FILE   = f'trn_shapes-colors-symbols_{which_label}.csv'
        NODE_COLORS = ['blue']*SHAPE_DIM + ['green']*COLOR_DIM + \
            ['gold']*SYMBOL_DIM + ['red']
        NUM_DATA    = SHAPE_DIM + COLOR_DIM + SYMBOL_DIM
        NUM_TOTAL   = SHAPE_DIM + COLOR_DIM + SYMBOL_DIM + NUM_LABELS
    else:
        raise ValueError(f'Invalid label! Got {which_label}')
    
    # --- create the figure directory if it does not exist ---

    if save_plots:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    # --- load the data ---

    df = pd.read_csv(DATA_DIR + DATA_FILE)
    array = df.to_numpy()

    # --- encode prior knowledge ---

    forbidden_edges = []
    # no edges between task nodes
    for i in range(NUM_DATA,NUM_TOTAL):
        for j in range(NUM_TOTAL):
            forbidden_edges.append((i,j))
    # no edges between nodes of the same domain
    for i in range(SHAPE_DIM):
        for j in range(SHAPE_DIM):
            forbidden_edges.append((i,j))
    for i in range(SHAPE_DIM, SHAPE_DIM+COLOR_DIM):
        for j in range(SHAPE_DIM, SHAPE_DIM+COLOR_DIM):
            forbidden_edges.append((i,j))
    for i in range(SHAPE_DIM+COLOR_DIM, NUM_DATA):
        for j in range(SHAPE_DIM+COLOR_DIM, NUM_DATA):
            forbidden_edges.append((i,j))

    prior_knowledge_array = -1*np.ones((NUM_TOTAL,NUM_TOTAL))
    for edge in forbidden_edges:
        prior_knowledge_array[edge[0],edge[1]] = 0

    if verbose:
        print('\nPrior knowledge:')
        print(prior_knowledge_array)

    # --- convert binary labels to {-1,1} ---

    if which_label == 'isSpeedLimit':
        array[:,-1] = 2*array[:,-1]-1
    else:
        array[:,NUM_DATA:] = 2*array[:,NUM_DATA:]-1

    # --- standardize the other rows ---

    if verbose:
        print('means:', np.mean(array[:,:NUM_DATA], axis=0))
        print('stds:', np.std(array[:,:NUM_DATA], axis=0))

    array[:,:NUM_DATA] = (array[:,:NUM_DATA] - np.mean(array[:,:NUM_DATA], axis=0)) / \
        np.std(array[:,:NUM_DATA], axis=0)

    # --- causal discovery ---

    alg = DirectLiNGAM(prior_knowledge=prior_knowledge_array,
                        measure=MEASURE, random_state=SEED)
    alg.fit(array)

    pred_dag = alg.adjacency_matrix_
    soft_dag = np.copy(pred_dag)
    pred_dag = (np.abs(pred_dag) > 0.0).astype(int)

    if verbose: 
        print('\nPredicted adjacency matrix:')
        print(pred_dag)
        print(np.round(soft_dag,2))

    # --- report some interesting info ---

    if verbose:
        print()
        cyc_mat = pred_dag.copy()
        cycle_nodes = []
        for i in range(2,NUM_TOTAL+1):
            cyc_mat = cyc_mat @ pred_dag
            for j in range(NUM_TOTAL):
                if cyc_mat[j,j] and (j not in cycle_nodes):
                    cycle_nodes.append(j)
                    print(f'Cycle at node {j}! {i} steps')
        if len(cycle_nodes) == 0:
            print('No cycles found!')

        cyc_mat = pred_dag.copy()
        for i in range(2,NUM_TOTAL):
            cyc_mat = cyc_mat @ pred_dag
            if np.sum(cyc_mat) == 0:
                print(f'Longest path in the graph: {i-1} steps')
                break

    # --- plotting the graph and matrix ---

    # graph
    g_pred = nx.DiGraph(pred_dag)
    plt.figure(figsize=(4,3))
    nx.draw(
        G=g_pred,
        node_color=NODE_COLORS,
        node_size=1000,
        arrowsize=6,
        with_labels=True,
        font_color='white',
        font_size=14,
        pos=nx.circular_layout(g_pred)
    )
    if save_plots:
        plt.savefig(SAVE_DIR + SAVE_FILE + '-graph.png')

    # colored adjacency matrix
    _, ax1 = plt.subplots(figsize=(4, 3), ncols=1)

    color_dag = np.ones((NUM_TOTAL,NUM_TOTAL,3)).astype(int)*255
    color_dag[0:2,:,[0,1]]      -= 55
    color_dag[2:5,:,[0,2]]      -= 55
    color_dag[5:NUM_DATA,:,2]   -= 55
    if which_label == 'isSpeedLimit':
        color_dag[-1,:,[1,2]]       -= 55
    else:
        color_dag[-4:,:,[1,2]]     -= 55
    color_dag[:,0:2,[0,1]]      -= 55
    color_dag[:,2:5,[0,2]]      -= 55
    color_dag[:,5:NUM_DATA,2]   -= 55
    if which_label == 'isSpeedLimit':
        color_dag[:,-1,[1,2]]       -= 55
    else:
        color_dag[:,-4:,[1,2]]    -= 55

    color_dag[pred_dag==1,:] -= 145

    ax1.set_title('est_graph')
    ax1.imshow(color_dag)

    size = pred_dag.shape[0]
    ticks = np.array(range(size))
    if which_label in ['shapes', 'colors']:
        tick_labels = ['s0', 's1', 'c0', 'c1', 'c2', 'u0', 'u1', 'u2', 'u3']
    elif which_label == 'isSpeedLimit':
        tick_labels = ['s0', 's1', 'c0', 'c1', 'c2'] + \
            [f'y{i}' for i in range(SYMBOL_DIM)] + ['u']
    ax1.set_xticks(ticks, tick_labels)
    ax1.set_yticks(ticks, tick_labels)

    secondary_ticks = np.array(range(size-1))+0.5
    ax1.set_xticks(secondary_ticks, minor=True)
    ax1.set_yticks(secondary_ticks, minor=True)
    ax1.grid(which='minor')
    if save_plots:
        plt.savefig(SAVE_DIR + SAVE_FILE + '-adj.png')

    # heat maps of soft labels
    fig, ax2 = plt.subplots(figsize=(4, 3), ncols=1)

    cmap = plt.get_cmap('seismic')
    norm = Normalize(vmin=-1.0, vmax=1.0)

    ax2.set_title('soft_graph')
    cax2 = ax2.imshow(soft_dag, cmap=cmap, norm=norm)

    ax2.set_xticks(ticks, tick_labels)
    ax2.set_yticks(ticks, tick_labels)

    secondary_ticks = np.array(range(size-1))+0.5
    ax2.set_xticks(secondary_ticks, minor=True)
    ax2.set_yticks(secondary_ticks, minor=True)
    ax2.grid(which='minor')
    fig.colorbar(cax2, ax=ax2, orientation='vertical')
    
    if save_plots:
        plt.savefig(SAVE_DIR + SAVE_FILE + '-heat.png')

    if show_plots:
        plt.show()

    return soft_dag

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    WHICH_LABELS = ['shapes', 'colors', 'isSpeedLimit']

    SAVE_PLOTS = False
    SHOW_PLOTS = False
    VERBOSE = False

    for which_label in WHICH_LABELS:
        soft_dag = main(which_label, SAVE_PLOTS, SHOW_PLOTS, VERBOSE)

        idx = np.where(np.abs(soft_dag) > 0.0)
        hard_dag = np.zeros_like(soft_dag)
        hard_dag[idx] = 1.0
        print(f'{which_label} hard_dag:')
        print(hard_dag)

# ------------------------------------------------------------------------------
