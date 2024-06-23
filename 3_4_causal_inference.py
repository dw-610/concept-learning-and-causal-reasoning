"""
This script tests out the causal learning framework on the sign data.

It looks to infer the causal effect of the CS shapes/color dimensions on the
binary one-hot shape/color classification variables of the signs.

In this script, DML is implemented to handle the effect of continuous treatment
variables.
"""

# ------------------------------------------------------------------------------
# imports

import os
import warnings
import pandas as pd
import networkx as nx
import numpy as np

from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from modules.causal_cs.graphs import graphs

# ------------------------------------------------------------------------------

def main(treatment, outcome, task):

    # --- constants ---

    which_label     = task

    data_dir        = 'local/causal_data/'
    if task in ['shapes', 'colors']:
        which_data      = 'sc'
        data_file       = f'trn_shapes-colors_{which_label}.csv'
    elif task == 'isSpeedLimit':
        which_data      = 'scy'
        data_file       = f'trn_shapes-colors-symbols_{which_label}.csv'
    else:
        raise ValueError(f'Invalid task: {task}')

    SHOW_GRAPH      = False
    SHOW_ESTIMAND   = False
    SHOW_ESTIMATE   = False

    # --- load the data ---

    df = pd.read_csv(data_dir + data_file)
    if task in ['shapes', 'colors']:
        for i in range(4):
            df[f'u{i}'] = df[f'u{i}'].astype(int)
    elif task == 'isSpeedLimit':
        df['u'] = df['u'].astype(int)
    else:
        raise ValueError(f'Invalid task: {task}')

    # --- load the graph ---

    try:
        graph_info = graphs[f'{which_data}_{which_label}']
    except KeyError:
        raise ValueError('Chosen graph not defined: ' + \
                         f'{which_data}_{which_label}')
    adj_mat = graph_info['adj_mat']
    node_names = graph_info['node_names']

    graph = nx.DiGraph(adj_mat)
    graph = nx.relabel_nodes(
        graph,
        {i: node_names[i] for i in range(len(node_names))}
    )
    if (treatment, outcome) not in graph.edges:
        return None, None, None, None, None

    graph = "\n".join(nx.generate_gml(graph))

    # --- causal inference ---

    model = CausalModel(
        data = df,
        treatment = treatment,
        outcome = outcome,
        graph = graph
    )

    if SHOW_GRAPH: model.view_model()

    estimand = model.identify_effect()
    if SHOW_ESTIMAND: print(estimand)

    estimate = model.estimate_effect(
        identified_estimand=estimand,
        method_name='backdoor.econml.dml.DML',
        target_units='ate',
        method_params={
            'init_params': {
                'model_y': RandomForestClassifier(),
                'model_t': RandomForestRegressor(),
                'model_final': RandomForestRegressor(),
                'discrete_outcome': True
            },
            'fit_params': {}
        }
    )

    if SHOW_ESTIMATE:
        print(f'Estimated avg. effect: {estimate.value}\n')

    # --- refutation ---

    refute_estimates = {}
    refute_p_values = {}
    passed_refutations = {}

    # common cause refutation
    refute_common_cause = model.refute_estimate(
        estimand=estimand,
        estimate=estimate,
        method_name='random_common_cause',
        show_progress_bar=False,
        num_simulations=90,
        n_jobs=18
    )
    refute_estimates['common_cause'] = refute_common_cause.new_effect
    refute_p_values['common_cause'] = \
        refute_common_cause.refutation_result['p_value'][0]
    passed_refutations['common_cause'] = not \
        refute_common_cause.refutation_result['is_statistically_significant']

    # subset refutation
    refute_subset = model.refute_estimate(
        estimand=estimand,
        estimate=estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.5,
        num_simulations=90,
        show_progress_bar=False,
        n_jobs=18
    )
    refute_estimates['subset'] = refute_subset.new_effect
    refute_p_values['subset'] = refute_subset.refutation_result['p_value'][0]
    passed_refutations['subset'] = not refute_subset.refutation_result[
        'is_statistically_significant'][0]

    return estimate.value, estimate.cate_estimates.ravel(), \
              refute_estimates, passed_refutations, refute_p_values
    
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    MUTE_WARNINGS   = True

    if MUTE_WARNINGS:
        os.environ['PYTHONWARNINGS'] = 'ignore'
        warnings.filterwarnings('ignore')

    TASKS = ['shapes', 'colors']
    DATA_VARS = ['s0', 's1', 'c0', 'c1', 'c2']
    TASK_VARS = ['u0', 'u1', 'u2', 'u3']

    for task in TASKS:
        effect_mat = np.zeros((len(DATA_VARS), len(TASK_VARS)))
        cc_passed_mat = np.zeros((len(DATA_VARS), len(TASK_VARS)))
        cc_refute_mat = np.zeros((len(DATA_VARS), len(TASK_VARS)))
        cc_p_val_mat  = np.zeros((len(DATA_VARS), len(TASK_VARS)))
        ss_passed_mat = np.zeros((len(DATA_VARS), len(TASK_VARS)))
        ss_refute_mat = np.zeros((len(DATA_VARS), len(TASK_VARS)))
        ss_p_val_mat  = np.zeros((len(DATA_VARS), len(TASK_VARS)))
        for i, treatment in enumerate(DATA_VARS):
            for j, outcome in enumerate(TASK_VARS):
                print(f'Task: {task}, treatment: {treatment}, outcome: {outcome}')
                effect, effects, refute_effects, passed_refutes, refute_p_vals = main(
                    treatment=treatment, outcome=outcome, task=task)
                effect_mat[i, j] = effect if effect is not None else np.nan
                cc_passed_mat[i, j] = passed_refutes['common_cause'] if passed_refutes is not None else np.nan
                cc_refute_mat[i, j] = refute_effects['common_cause'] if refute_effects is not None else np.nan
                cc_p_val_mat[i, j] = refute_p_vals['common_cause'] if refute_p_vals is not None else np.nan
                ss_passed_mat[i, j] = passed_refutes['subset'] if passed_refutes is not None else np.nan
                ss_refute_mat[i, j] = refute_effects['subset'] if refute_effects is not None else np.nan
                ss_p_val_mat[i, j] = refute_p_vals['subset'] if refute_p_vals is not None else np.nan
        print(f'\nTask: {task}')
        print('Effect matrix:')
        print(effect_mat)
        print('Common cause passed matrix:')
        print(cc_passed_mat)
        print('Common cause refute matrix:')
        print(cc_refute_mat)
        print('Common cause p-value matrix:')
        print(cc_p_val_mat)
        print('Subset passed matrix:')
        print(ss_passed_mat)
        print('Subset refute matrix:')
        print(ss_refute_mat)
        print('Subset p-value matrix:')
        print(ss_p_val_mat)

    TASK = 'isSpeedLimit'
    DATA_VARS = ['s0', 's1', 'c0', 'c1', 'c2', 'y0', 'y1', 'y2', 'y3', 'y4',
                 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13', 'y14']
    TASK_VAR = ['u']

    effect_mat = np.zeros((len(DATA_VARS), len(TASK_VAR)))
    cc_passed_mat = np.zeros((len(DATA_VARS), len(TASK_VAR)))
    cc_refute_mat = np.zeros((len(DATA_VARS), len(TASK_VAR)))
    cc_p_val_mat  = np.zeros((len(DATA_VARS), len(TASK_VAR)))
    ss_passed_mat = np.zeros((len(DATA_VARS), len(TASK_VAR)))
    ss_refute_mat = np.zeros((len(DATA_VARS), len(TASK_VAR)))
    ss_p_val_mat  = np.zeros((len(DATA_VARS), len(TASK_VAR)))
    for i, treatment in enumerate(DATA_VARS):
        for j, outcome in enumerate(TASK_VAR):
            print(f'Task: {TASK}, treatment: {treatment}, outcome: {outcome}')
            effect, effects, refute_effects, passed_refutes, refute_p_vals = main(
                treatment=treatment, outcome=outcome, task=TASK)
            effect_mat[i, j] = effect if effect is not None else np.nan
            cc_passed_mat[i, j] = passed_refutes['common_cause'] if passed_refutes is not None else np.nan
            cc_refute_mat[i, j] = refute_effects['common_cause'] if refute_effects is not None else np.nan
            cc_p_val_mat[i, j] = refute_p_vals['common_cause'] if refute_p_vals is not None else np.nan
            ss_passed_mat[i, j] = passed_refutes['subset'] if passed_refutes is not None else np.nan
            ss_refute_mat[i, j] = refute_effects['subset'] if refute_effects is not None else np.nan
            ss_p_val_mat[i, j] = refute_p_vals['subset'] if refute_p_vals is not None else np.nan
    print(f'\nTask: {TASK}')
    print('Effect matrix:')
    print(effect_mat)
    print('Common cause passed matrix:')
    print(cc_passed_mat)
    print('Common cause refute matrix:')
    print(cc_refute_mat)
    print('Common cause p-value matrix:')
    print(cc_p_val_mat)
    print('Subset passed matrix:')
    print(ss_passed_mat)
    print('Subset refute matrix:')
    print(ss_refute_mat)
    print('Subset p-value matrix:')
    print(ss_p_val_mat)

# ------------------------------------------------------------------------------