"""
This script will evaluate an already trained domain learner.
"""

# ------------------------------------------------------------------------------
# imports



# ------------------------------------------------------------------------------

def main():

    # # --- evaluation ---

    if domain == 'shapes':
        legend = ['triangle', 'square', 'octagon', 'circle']
    elif domain == 'colors':
        legend = ['red', 'blue', 'yellow', 'black']
        colors = ['#FF0000', '#0000FF', '#DDDD00', '#000000']

    ctrl.eval_plot_accuracy_curves(save_path=f'{figpath}{domain}_acc.png')

    # ctrl.eval_plot_scattered_features(
    #     legend=legend,
    #     save_path=f'{figpath}{domain}_2Dfeats.png',
    #     colors=colors if domain == 'colors' else None
    # )

    ctrl.eval_plot_similarity_heatmap(legend=legend,
                                      save_path=f'{figpath}{domain}_simHeatmap')

    ctrl.eval_show_decoded_protos(
        legend=legend,
        save_path=f'{figpath}{domain}_dec_protos.png'
    )

    ctrl.eval_compare_true_and_generated(which='training')

    ctrl.eval_compare_true_and_generated(
        which='validation',
        save_path=f'{figpath}{domain}_trueVsGen.png'
    )

    ctrl.eval_visualize_all_dimensions(
        save_path=f'{figpath}{domain}_visDims.png',
        fixed_dims=[0.0, 0.0] if domain == 'shapes' else [0.0, 0.0, 0.0]
    )

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------