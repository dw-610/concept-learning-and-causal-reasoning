"""
This module contains CATE matrices corresponding to different DAGs.

Naming convention: <data>_<task>_<method>

Data options:
- sc: shapes and colors

Task options:
- shapes: shape classification
- colors: color classification

Method options:
- hand: hand-crafted
- modpc: modified PC algorithm
- full: full bipartite graph
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np

# ------------------------------------------------------------------------------

# # this one found with the old dataset
# sc_shapes_hand = np.array([
#     [-0.00095171, -0.02874802,  0.01221106,  0.18827822],
#     [ 0.11056914, -0.09437577, -0.20372708,  0.10151358],
#     [     np.nan,      np.nan,      np.nan,      np.nan],
#     [     np.nan,      np.nan,      np.nan,      np.nan],
#     [     np.nan,      np.nan,      np.nan,      np.nan]
# ])

sc_shapes_hand = np.array([
    [-0.02249366, -0.17901014, -0.01543077,  0.10812563],
    [-0.16754752,  0.06228233,  0.06852698, -0.63536288],
    [     np.nan,      np.nan,      np.nan,      np.nan],
    [     np.nan,      np.nan,      np.nan,      np.nan],
    [     np.nan,      np.nan,      np.nan,      np.nan]
])

# ------------------------------------------------------------------------------

# # this one found with the old dataset
# sc_shapes_modpc = np.array([
#     [-2.26788092e-01, -5.84469834e-02,          np.nan,  3.51144967e-01],
#     [ 2.28000000e-02,  0.00000000e+00, -1.72050673e-04, -3.30000000e-02],
#     [         np.nan,          np.nan,          np.nan,          np.nan],
#     [         np.nan,          np.nan,          np.nan,          np.nan],
#     [-2.67974980e-02,  1.45651956e-01,          np.nan,          np.nan]
# ])

sc_shapes_modpc = np.array([
    [-6.2300000e-01,         np.nan,         np.nan, -2.7690000e-01],
    [-6.3750000e-01,         np.nan, -2.7200000e-06, -1.1200000e-02],
    [        np.nan,         np.nan,         np.nan,         np.nan],
    [        np.nan, -2.8000000e-03,         np.nan,         np.nan],
    [        np.nan,  1.0969202e-01,         np.nan,         np.nan]
])

# ------------------------------------------------------------------------------

# # this one found with the old dataset
# sc_shapes_full = np.array([
#     [-0.18683574, -0.00795793, -0.00674751,  0.17071002],
#     [ 0.13246535, -0.01006058, -0.01174088, -0.1875961 ],
#     [-0.06300383, -0.00099730,  0.01153268,  0.05488206],
#     [-0.00725679,  0.02294648,  0.02361301,  0.01000408],
#     [ 0.04144396,  0.12050451,  0.00961328, -0.10502646]
# ])

sc_shapes_full = np.array([
    [-0.05966724, -0.01087648, -0.02465799,  0.04628466,],
    [-0.06161005,  0.02506466,  0.16406046, -0.13464518,],
    [-0.00054033,  0.00635560,  0.00273285, -0.01183058,],
    [-0.00629095, -0.00623490,  0.00254027,  0.00475268,],
    [-0.00251965,  0.01286366, -0.00299027, -0.07561021,]
])

# ------------------------------------------------------------------------------

# # this one found with the old dataset
# sc_colors_hand = np.array([
#     [     np.nan,      np.nan,      np.nan,      np.nan],
#     [     np.nan,      np.nan,      np.nan,      np.nan],
#     [-0.00721046, -0.01727304, -0.04428178,  0.03154084],
#     [-0.00103767, -0.07802185,  0.04181605, -0.00102601],
#     [-0.37508736,  0.07843075,  0.22657112,  0.02530879]
# ])

sc_colors_hand = np.array([
    [     np.nan,      np.nan,      np.nan,      np.nan],
    [     np.nan,      np.nan,      np.nan,      np.nan],
    [ 0.01167141, -0.08799358,  0.02463661,  0.02436871],
    [ 0.07690195, -0.02452484, -0.01548146, -0.04330827],
    [ 0.40256121, -0.03291514,  0.04751875, -0.00225816]
])

# ------------------------------------------------------------------------------

# # this one found with the old dataset
# sc_colors_modpc = np.array([
#     [     np.nan,  0.00784336,  0.0607039 ,      np.nan],
#     [     np.nan,      np.nan,  0.06162325,      np.nan],
#     [-0.00754966, -0.03803232,      np.nan,  0.01654861],
#     [-0.00452181, -0.12097188,      np.nan, -0.00371599],
#     [-0.20871372,  0.02391861,  0.01929307,  0.04615306],
# ])

sc_colors_modpc = np.array([
    [     np.nan,      np.nan,      np.nan,      np.nan,],
    [     np.nan,      np.nan,      np.nan,      np.nan,],
    [ 0.03168947, -0.04250940,      np.nan,  0.04580827,],
    [ 0.02210809, -0.02767857,      0.0000, -0.03684040,],
    [     np.nan, -0.04960818,      0.0000,      np.nan,]
])

# ------------------------------------------------------------------------------

# # this one found with the old dataset
# sc_colors_full = np.array([
#     [-0.00895748,  0.00349816, -0.00430417,  0.00112074],
#     [ 0.02127216, -0.00389572, -0.00429219, -0.00250268],
#     [-0.00713055, -0.02028008, -0.00232162,  0.09995194],
#     [ 0.00589910, -0.02855450,  0.03496987, -0.00064432],
#     [-0.14237551,  0.00329908,  0.07707765,  0.06100112]
# ])

sc_colors_full = np.array([
    [ 0.01775798, -0.04274665, -0.01749595,  0.00686142],
    [-0.03839930,  0.03407104,  0.09147013,  0.00051941],
    [ 0.00244756, -0.06439932,  0.00066923,  0.03646392],
    [ 0.07348202, -0.02601585, -0.00166067, -0.02985662],
    [-0.00301922, -0.00335106,  0.11051865,  0.00128194]
])

# ------------------------------------------------------------------------------

scy_isSpeedLimit_lingam = np.array([
    [ 0.003],
    [-0.007],
    [ 0.   ],
    [ 0.007],
    [ 0.013],
    [-0.695],
    [-0.864],
    [ 0.356],
    [ 0.   ],
    [-0.167],
    [-0.425],
    [np.nan],
    [np.nan],
    [ 0.24 ],
    [np.nan]
])

# scy_isSpeedLimit_lingam = np.array([
#     [ 0.003],
#     [-0.007],
#     [ 0.   ],
#     [ 0.007],
#     [ 0.013],
#     [-0.864],
#     [-0.695],
#     [ 0.356],
#     [ 0.   ],
#     [-0.167],
#     [-0.425],
#     [np.nan],
#     [np.nan],
#     [ 0.24 ],
#     [np.nan]
# ])

# ------------------------------------------------------------------------------

mats = {
    'sc_shapes_hand': sc_shapes_hand,
    'sc_shapes_modpc': sc_shapes_modpc,
    'sc_shapes_full': sc_shapes_full,
    'sc_colors_hand': sc_colors_hand,
    'sc_colors_modpc': sc_colors_modpc,
    'sc_colors_full': sc_colors_full,
    'scy_isSpeedLimit_lingam': scy_isSpeedLimit_lingam,
}
"""
Dictionary of effect matrices.
"""

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    mats = {
        # 'sc_shapes_hand': sc_shapes_hand,
        # 'sc_shapes_modpc': sc_shapes_modpc,
        # 'sc_shapes_full': sc_shapes_full,
        # 'sc_colors_hand': sc_colors_hand,
        # 'sc_colors_modpc': sc_colors_modpc,
        # 'sc_colors_full': sc_colors_full,
        'scy_isSpeedLimit_lingam': scy_isSpeedLimit_lingam,
    }
    
    # xlabels = ['u0', 'u1', 'u2', 'u3']
    xlabels = ['u']
    ylabels = ['s0', 's1', 'c0', 'c1', 'c2', 'y0', 'y1', 'y2', 'y3', 'y4', 
               'y5', 'y6', 'y7', 'y8', 'y9']
    
    for name, mat in mats.items():
        mat = np.nan_to_num(mat)
        n_mat = np.abs(mat)
        n_mat = n_mat / np.max(n_mat)
        row_sums = np.sum(n_mat, axis=1)

        print(f'\n{name}:')
        print()
        print(np.round(mat, 3))
        print()
        print(np.round(n_mat, 3))
        print()
        print(np.round(row_sums, 3))

        plt.figure(figsize=(3, 3))
        plt.imshow(n_mat, cmap='inferno')
        plt.xticks(range(len(xlabels)), xlabels)
        plt.yticks(range(len(ylabels)), ylabels)
        plt.colorbar()
        plt.show()