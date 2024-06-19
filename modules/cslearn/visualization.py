"""
This file contains high-level plotting calls for evaluating the learners and 
visualizing different aspects of the results.
"""

# ------------------------------------------------------------------------------

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Optional

from .arch.layers import ReparameterizationLayer

from . import utilities as utils

# ------------------------------------------------------------------------------

class PrototypePlotter2D():
    """
    Class for plotting the learned prototypes throughout the domain learning
    process. Only works when the feature space is two-dimensional.
    Used to obtain plots *during training*.
    """
    def __init__(
            self,
            initial_prototypes: np.array,
            save_directory: str,
            dataset: str,
            show: bool = False,
            colors: Optional[list] = None,
            legend: Optional[list] = None
        ):
        """
        Parameters
        ----------
        initial_prototypes : int
            Numpy array containing the initial prototype points.
            Rows represent prototypes, columns represent 2D features.
        save_directory : str
            Path to the directory where plots are to be saved.
        dataset : str
            A string describing the dataset being used (e.g. MNIST)
        show : bool, optional
            Set this to True to show the plot during training.
            Default value is False.
        colors : list, optional
            A list of color-specifying strings, or None.
            If None, colors are automatically generated.
            Default value is None.
        legend : list, optional
            A list of strings to use as the legend entries, or None.
            If None, no legend is used.
            Default value is None.
        """

        # make assertions and set attributes
        self.show = show
        self.dataset = dataset
        self.save_dir = save_directory

        assert initial_prototypes.shape[1] == 2, \
            "Feature space must be two-dimensional to use the plotter."

        self.n = len(initial_prototypes)

        if isinstance(colors, list):
            assert len(colors) == self.n, \
                "Length of colors must match the length of initial_prototypes"
            
        if isinstance(legend, list):
            assert len(legend) == self.n, \
                "Length of legend must match the length of initial_prototypes"

        if colors is None:
            cm_subsection = np.linspace(0, 1, self.n)
            self.colors = [cm.jet(x) for x in cm_subsection]
        else:
            self.colors = colors

        # initialize plot objects and step counter
        self.fig, self.ax = plt.subplots()
        self.scatter_objects = []
        self.step = 0

        # plot the initial prototypes
        for i in range(self.n):
            scat = self.ax.scatter(
                initial_prototypes[i,0], 
                initial_prototypes[i,1], 
                color=self.colors[i], 
                marker='o'
            )
            self.scatter_objects.append(scat)

        if isinstance(legend, list):
            self.fig.legend(legend, loc='center right')
        self.ax.grid(visible=True)
        self.ax.set_title(f'{dataset} Prototypes: Step {self.step}')

        # set axis limits
        xmin = np.min(initial_prototypes[:,0])
        xmax = np.max(initial_prototypes[:,0])
        xlow = 0.9*xmin if xmin > 0 else 1.1*xmin
        xhigh = 1.1*xmax if xmax > 0 else 0.9*xmax
        self.ax.set_xlim((xlow,xhigh))

        ymin = np.min(initial_prototypes[:,1])
        ymax = np.max(initial_prototypes[:,1])
        ylow = 0.9*ymin if ymin > 0 else 1.1*ymin
        yhigh = 1.1*ymax if ymax > 0 else 0.9*ymax
        self.ax.set_ylim((ylow,yhigh))

    def update_and_save(
            self,
            updated_protos: np.array
        ):
        """
        Method for updating and saving the prototype plot.

        Parameters
        ----------
        updated_protos : np.array
            Numpy array containing the updated prototypes.
        """
        
        self.step += 1

        # remove the old scatter points
        for scat in self.scatter_objects:
            scat.remove()
        self.scatter_objects.clear()

        # update the plot
        for i in range(self.n):
            scat = self.ax.scatter(
                updated_protos[i,0],
                updated_protos[i,1],
                color=self.colors[i],
                marker='o'
            )
            self.scatter_objects.append(scat)
        self.ax.set_title(f'{self.dataset} Prototypes: Step {self.step}')

        # adjust axis limits if needed
        if self.x_lim is not None:
            self.ax.set_xlim(self.x_lim)
        else:
            xmin = np.min(updated_protos[:,0])
            xmax = np.max(updated_protos[:,0])
            xlow = 0.9*xmin if xmin > 0 else 1.1*xmin
            xhigh = 1.1*xmax if xmax > 0 else 0.9*xmax
            self.ax.set_xlim((xlow,xhigh))

        if self.y_lim is not None:
            self.ax.set_ylim(self.y_lim)
        else:
            ymin = np.min(updated_protos[:,1])
            ymax = np.max(updated_protos[:,1])
            ylow = 0.9*ymin if ymin > 0 else 1.1*ymin
            yhigh = 1.1*ymax if ymax > 0 else 0.9*ymax
            self.ax.set_ylim((ylow,yhigh))

        # show the plot if specified, save to indicated directory
        if self.show:
            plt.pause(0.1)

        self.fig.savefig(
            f'{self.save_dir}/step_{self.step}',
            dpi=300,
            bbox_inches='tight'
        )

# ------------------------------------------------------------------------------

def plot_decoded_protos(
        decoder,
        prototypes,
        dataset = str,
        legend: Optional[list] = None,
        save_path: Optional[str] = None,
        show: Optional[bool] = True,
        block: Optional[bool] = False
    ):
    """
    Function to display the decoded images corresponding to the prototypes.

    Parameters
    ----------
    decoder : keras.models.Model
        A trained decoder model taking features to images.
    prototypes : np.array
        Numpy array of prototypes.
        Rows represent prototypes and columns represent features.
    dataset : str
        String identifier of the dataset used, e.g. 'mnist'
    legend : list, optional
        A list of strings to use as the legend entries.
        If None, no legend is used.
    save_path : str, optional
        Path to save image, including the file name. 
        If None, image is not saved.
        Default value is None.
    show : bool, optional
        Set this to True to show the plot.
        Default value is True.
    block : bool, optional
        Set this to True to block the program until the plot is closed.
    """

    num_protos = len(prototypes)

    grid_width = int(np.ceil(np.sqrt(num_protos)))
    grid_height = int(np.ceil(num_protos/grid_width))

    _, axs = plt.subplots(
        nrows=grid_width,
        ncols=grid_height,
        figsize=(grid_width*2,grid_height*2)
    )

    if dataset == 'mnist':
        ims = (decoder.predict(prototypes)+1)/2.0
        ims = np.clip(ims, 0, 1)
        ims = (ims*255).astype(np.uint8)
    else:
        ims = ((decoder.predict(prototypes)+1)/2.0).astype(np.float32)

    i = 0
    for ax_row in axs:
        if grid_height > 1:
            for ax in ax_row:
                if dataset == 'mnist':
                    ax.imshow(ims[i], cmap='gray', vmin=0, vmax=255)
                else:
                    ax.imshow(ims[i])
                ax.axis('off')
                ax.set_title(
                    legend[i] if legend is not None else f'Prototype {i+1}'
                )
                i+=1

                if i == num_protos:
                    break
            if i == num_protos:
                break
        else:
            if dataset == 'mnist':
                ax_row.imshow(ims[i], cmap='gray', vmin=0, vmax=255)
            else:
                ax_row.imshow(ims[i])
            ax_row.axis('off')
            ax_row.set_title(
                    legend[i] if legend is not None else f'Prototype {i+1}'
                )
            i+=1

            if i == num_protos:
                break

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(utils.get_unused_name(save_path))

    if show:
        plt.show(block=block)

# ------------------------------------------------------------------------------

def plot_true_and_decoded(
        data,
        labels,
        encoder,
        decoder,
        dataset = str,
        is_variational: Optional[bool] = False,
        save_path: Optional[str] = None,
        show: Optional[bool] = True,
        block: Optional[bool] = False
    ):
    """
    Function to plot 16 true images next to the predicted images.

    Parameters
    ----------
    data : 
        A batch of input data
    labels : 
        A batch of true output data.
        Made up of two outputs to be consistent with domain learner model.
        First element is the image labels, second element is class labels.
    encoder :
        The trained encoder model.
    decoder :
        The trained decoder model.
    dataset : str
        String identifier of the dataset used, e.g. 'mnist'
    is_variational : bool, optional
        Set this to True if the autoencoder model is variational.
    num_to_show : int, optional
        The number of images to show.
        Default value is 16.
    save_path : str, optional
        Path to save image, with the file name. 
        If None, image is not saved.
        Default value is None.
    show : bool, optional
        Set this to True to show the plot.
        Default value is True.
    block : bool, optional
        Set this to True to block the program until the plot is closed.
        Default value is False.
    """

    assert len(data) >= 16, \
        "num_to_show must be at most the batch size of data and labels"

    true_im = (labels[0:16] + 1)/2.0

    encoded = encoder.predict(data[0:16]).astype(np.float32)

    if is_variational:
        latent_dim = int(encoded.shape[-1]/2)
        features = ReparameterizationLayer(latent_dim)(encoded)[2].numpy()
        pred = decoder.predict(features).astype(np.float32)
    else:
        features = encoded
        pred = decoder.predict(features).astype(np.float32)
    
    pred_im = (pred + 1)/2.0
    pred_im = np.clip(pred_im, 0, 1)

    _, axes   = plt.subplots(4,8, figsize=(16,8))

    for i in range(4):
        for j in range(4):
            idx = 4*i + j

            if dataset == 'mnist':
                axes[i, 2 * j].imshow(
                    true_im[idx], cmap='gray', vmin=0, vmax=1
                )
                axes[i, 2 * j + 1].imshow(
                    pred_im[idx], cmap='gray', vmin=0, vmax=1
                )
            else:
                axes[i, 2 * j].imshow(true_im[idx])
                axes[i, 2 * j + 1].imshow(pred_im[idx])

            axes[i, 2 * j].axis('off')
            axes[i, 2 * j].set_title(f'True {idx+1}')

            axes[i, 2 * j + 1].axis('off')
            axes[i, 2 * j + 1].set_title(f'Pred {idx+1}')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(utils.get_unused_name(save_path))   

    if show:
        plt.show(block=block) 

# ------------------------------------------------------------------------------

def plot_scattered_features(
        features,
        labels,
        colors: Optional[list] = None,
        legend: Optional[list] = None,
        save_path: Optional[str] = None,
        show: Optional[bool] = True,
        block: Optional[bool] = False
    ):
    """
    This function plots the provided 3D features and colors them according to
    their labels.

    Parameters
    ----------
    features : np.array
        Array of features to be plotted. Must be 2D. Shape is (n,2).
    labels : np.array
        Array of labels for the features. Shape is (n, number_of_classes),
        where the labels are one-hot encoded.
    colors : list
        A list of color-specifying strings.
    legend : list   
        A list of strings to use as the legend entries.
    save_path : str, optional
        Path to save image, with the file name. 
        If None, image is not saved.
        Default value is None.
    show : bool, optional
        Set this to True to show the plot.
        Default value is True.
    block : bool, optional
        Set this to True to block the program until the plot is closed.
        Default value is False.
    """
    
    # get dimensionality of features
    dim = features.shape[1]

    # get the number of properties
    number_of_classes = labels.shape[1]

    # convert labels to integers from one-hot encoding
    labels = np.argmax(labels, axis=1)

    # create the figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d' if dim==3 else None)

    # loop through each class and plot the scatters
    for i in range(number_of_classes):
        subset = features[labels == i]

        # plot the features for the current property
        if dim == 2:
            ax.scatter(
                subset[:, 0], 
                subset[:, 1], 
                s = 100, 
                label=str(i),
                alpha=0.75,
                marker='.',
                color=colors[i] if colors is not None else None,
            )
        elif dim == 3:
            ax.scatter(
                subset[:, 0], 
                subset[:, 1], 
                subset[:, 2], 
                s = 100, 
                label=str(i),
                alpha=0.75,
                marker='.',
                color=colors[i] if colors is not None else None,
            )
        else:
            raise NotImplementedError("features must be 2D or 3D")

    # plot the legend if given
    if legend is not None:
        plt.legend(legend)

    # set axis labels and the title
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if dim==3:
        ax.set_zlabel('Feature 3')
    ax.set_title(f'{dim}D Feature Vectors')
    plt.grid(visible=True)

    if save_path is not None:
        save_path = utils.get_unused_name(save_path)
        plt.savefig(save_path)

    if show:
        plt.show(block=block)

# ------------------------------------------------------------------------------

def plot_scattered_prototypes(
        prototypes,
        colors: Optional[list] = None,
        legend: Optional[list] = None,
        save_path: Optional[str] = None,
        show: Optional[bool] = True,
        block: Optional[bool] = False
    ):
    """
    This function plots the 2D prototypes.

    Parameters
    ----------
    prototypes : np.array
        Array of prototypes.
    colors : list, optional
        A list of color-specifying strings.
        If None, colors are automatically generated.
        Default value is None.
    legend : list, optional
        A list of strings to use as the legend entries.
        If None, no legend is used.
        Default value is None.
    save_path : str, optional
        Path to save image, with the file name. 
        If None, image is not saved.
        Default value is None.
    show : bool, optional
        Set this to True to show the plot.
        Default value is True.
    block : bool, optional
        Set this to True to block the program until the plot is closed.
        Default value is False.
    """

    # get dimensionality of features
    dim = prototypes.shape[1]

    # get the number of properties
    number_of_classes = prototypes.shape[0]

    # create the figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d' if dim==3 else None)

    # loop through each class and plot the scatters
    for i in range(number_of_classes):
        if dim == 2:
            plt.scatter(
                prototypes[i, 0], 
                prototypes[i, 1], 
                label=str(i),
                alpha=0.75,
                marker='.',
                color=colors[i] if colors is not None else None,
                s = 100
            )
        elif dim == 3:
            ax.scatter(
                prototypes[i, 0], 
                prototypes[i, 1], 
                prototypes[i, 2], 
                label=str(i),
                alpha=0.75,
                marker='.',
                color=colors[i] if colors is not None else None,
                s = 100
            )
        else:
            raise ValueError("features must be 2D or 3D")

    # plot the legend if given
    if legend is not None:
        plt.legend(legend)

    # set axis labels and the title
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if dim==3:
        ax.set_zlabel('Feature 3')
    ax.set_title(f'{dim}D Prototype Vectors')
    plt.grid(visible=True)

    if save_path is not None:
        plt.savefig(utils.get_unused_name(save_path))

    if show:
        plt.show(block=block)

# ------------------------------------------------------------------------------

def plot_heatmap(
        matrix,
        title,
        legend,
        save_path: Optional[str] = None,
        show: Optional[bool] = True,
        block: Optional[bool] = False
    ):
    """
    This function plots a heatmap of the given matrix.

    Parameters
    ----------
    matrix : np.array
        Array of values to be plotted.
    title : str
        Title of the plot.
    legend : list
        A list of strings to use as the legend entries.
    save_path : str, optional
        Path to save image, with the file name. 
        If None, image is not saved.
        Default value is None.
    show : bool, optional
        Set this to True to show the plot.
        Default value is True.
    block : bool, optional
        Set this to True to block the program until the plot is closed.
        Default value is False.
    """

    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(legend)), legend, rotation='vertical')
    plt.yticks(np.arange(len(legend)), legend)
    plt.title(title)

    if save_path is not None:
        plt.savefig(utils.get_unused_name(save_path))

    if show:
        plt.show(block=block)

# ------------------------------------------------------------------------------
        
def visualize_dimension(
        features: np.array,
        decoder,
        dim: int,
        min_val: float,
        max_val: float,
        steps: int,
        fixed_dims: Optional[list] = None,
        is_random_fixed_dims: bool = False,
        is_grayscale: bool = False,
        show: Optional[bool] = True,
        save_path: Optional[str] = None,
        block: Optional[bool] = False
    ):
    """
    This method visualizes the feature space along a specified dimension.
    Other dimensions are fixed; if not specified, they are fixed at zero.

    Parameters
    ----------
    features : np.array
        Array of features to be plotted.
    decoder : keras.models.Model
        A trained decoder model taking features to images.
    dim : int
        The dimension to visualize along.
        Dimensions are indexed from 1 to n_features.
    min_val : float
        The minimum value of the dimension, starting point.
    max_val : float
        The maximum value of the dimension, ending point.
    steps : int
        The number of steps to take between min_val and max_val.
    fixed_dims : list
        A list of dimensions to fix. 
        If specified, list should be of length n_features - 1.
        If not specified, all other dimensions are either fixed at their
            respective means, or are randomly chosen if is_random_fixed_dims 
            is True.
        Default value is None.
    is_random_fixed_dims : bool
        If True, the other dimensions will be randomly chosen.
        If False, the other dimensions will be fixed at zero.
        Default value is False.
    is_grayscale : bool
        If True, the images will be plotted using grayscale.
        If False, images will be plotted in color.
        Default value is False.
    show : bool
        If True, the plot will be shown.
        If False, the plot will not be shown.
        Default value is True.
    save_path : str
        If specified, the plot will be saved to the specified path.
        Default value is None.
    """

    n_dim = features.shape[1]
        
    # create a list of values for the specified dimension
    vals = np.linspace(min_val, max_val, steps)

    # create a list of values for the fixed dimensions
    if fixed_dims is None:
        fixed_dims = []
        # generate if random chosen, otherwise set at the mean
        if is_random_fixed_dims:
            for i in range(n_dim):
                if i == dim-1:
                    continue
                fmax = np.max(features[:, i])
                fmin = np.min(features[:, i])
                fixed_dims.append(np.random.uniform(low=fmin, high=fmax))
        else:
            for i in range(n_dim):
                if i == dim-1:
                    continue
                mean = np.mean(features[:, i])
                fixed_dims.append(mean)

    # insert a zero at the dimensions to visualize
    fixed_dims.insert(dim-1, 0)

    # create an array of all the points to visualize
    points = np.zeros((steps, n_dim))
    for i in range(steps):
        points[i, :] = fixed_dims
        points[i, dim-1] = vals[i]

    # decode the features
    decoded = decoder(points).numpy()

    plt.figure(figsize=(steps, .86), dpi=300)
    for i in range(steps):
        plt.subplot(1, steps, i+1)
        if is_grayscale:
            image = (decoded[i,:,:,:]+1.0)/2.0
            image = np.clip(image, 0, 1)
            image = (image*255).astype(np.uint8)
            plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow((decoded[i, :, :, :]+1.0)/2.0)
        plt.axis('off')
        plt.xlabel(str(points[i, :]))
    # remove white space between plots
    plt.subplots_adjust(wspace=0, top=1, bottom=0, left=0, right=1)

    if save_path is not None:
        plt.savefig(utils.get_unused_name(save_path), dpi=300)

    if show:
        plt.show(block=block)

# ------------------------------------------------------------------------------
        
def visualize_all_dimensions(
        features: np.array,
        decoder,
        steps: int,
        fixed_dims: Optional[list] = None,
        is_random_fixed_dims: bool = False,
        is_grayscale: bool = False,
        show: Optional[bool] = True,
        save_path: Optional[str] = None,
        block: Optional[bool] = False
    ):
    """
    This method plots a grid of images to visualize all the dimensions at
    once. Each row corresponds to a dimension, and each column corresponds
    to a different value for that dimension. For a given row, all other
    dimensions are held fixed at either their mean value or a random value.

    Parameters
    ----------
    features : np.array
        Array of features - used to max and min values for each dimension.
    decoder : tf.keras.models.Model
        A trained decoder model taking features to images.
    steps : int
        The number of steps to take between min_val and max_val.
    fixed_dims : list
        A list of dimensions to fix.
        If specified, list should be of length n_features - 1.
        If not specified, all other dimensions are either fixed at their
            respective means, or are randomly chosen if is_random_fixed_dims
            is True.
        Default value is None.
    is_random_fixed_dims : bool
        If True, the other dimensions will be randomly chosen.
        If False, the other dimensions will be fixed at zero.
        Default value is False.
    is_grayscale : bool
        If True, the images will be plotted using grayscale.
        If False, images will be plotted in color.
        Default value is False.
    show : bool
        If True, the plot will be shown.
        If False, the plot will not be shown.
        Default value is True.
    save_path : str
        If specified, the plot will be saved to the specified path.
        Default value is None.
    block : bool
        If True, the program will be blocked until the plot is closed.
        If False, the program will not be blocked.
        Default value is False.
    """

    n_dim = features.shape[1]

    # initialize the subplots
    fig, axs = plt.subplots(n_dim, steps, figsize=(steps, n_dim))

    # if not provided, set the fixed dimension values
    if fixed_dims is None:
        fixed_dims = []
        # generate if random chosen, otherwise set at the mean
        if is_random_fixed_dims:
            for i in range(n_dim):
                fmax = np.max(features[:, i])
                fmin = np.min(features[:, i])
                fixed_dims.append(np.random.uniform(low=fmin, high=fmax))
        else:
            for i in range(n_dim):
                mean = np.mean(features[:, i])
                fixed_dims.append(mean)

    print('Fixed: %s' % str(fixed_dims))

    for dim in range(1, n_dim+1):

        min_val = np.min(features[:, dim-1])
        max_val = np.max(features[:, dim-1])

        print('Dimension: %d - Min: %.3f - Max: %.3f'
                % (dim, min_val, max_val))
        
        # create a list of values for the specified dimension
        vals = np.linspace(min_val, max_val, steps)

        # create an array of all the points to visualize
        points = np.zeros((steps, n_dim))
        for i in range(steps):
            points[i, :] = fixed_dims
            points[i, dim-1] = vals[i]

        # decode the features
        decoded = decoder(points).numpy()

        # plot the decoded images in the row
        for i in range(steps):
            if is_grayscale:
                image = (decoded[i,:,:,:]+1.0)/2.0
                image = np.clip(image, 0, 1)
                image = (image*255).astype(np.uint8)
                axs[dim-1, i].imshow(
                    image,
                    cmap='gray',
                    vmin=0,
                    vmax=255
                )
            else:
                image = (decoded[i,:,:,:]+1.0)/2.0
                image = np.clip(image, 0, 1)
                axs[dim-1, i].imshow(image)
            axs[dim-1, i].axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(n_dim):
        axs[i, 0].set_ylabel(f'Dim {i+1}')

    if save_path is not None:
        plt.savefig(utils.get_unused_name(save_path), dpi=300)

    if show:
        plt.show(block=block)

# ------------------------------------------------------------------------------
        
def plot_similarity_histograms( 
        features: np.array,
        labels: np.array,
        number_of_properties: int,
        legend: list,
        show: Optional[bool] = True,
        save_path: Optional[str] = None,
        block: Optional[bool] = False,
        similarity_c: Optional[float] = 1.0
    ):
    """
    This method plots a grid of similarity histograms for pairs of
    properties. Each histogram shows the distribution of similarities
    between the features of two properties.

    Parameters
    ----------
    features : np.array
        Array of features to be plotted.
    labels : np.array
        Array of labels for the features.
    number_of_properties : int
        The number of properties.
    legend : list
        A list of strings to use as the legend entries.
    show : bool
        If True, the plot will be shown.
        If False, the plot will not be shown.
        Default value is True.
    save_path : str
        If specified, the plot will be saved to the specified path.
        Default value is None.
    block : bool
        If True, the program will be blocked until the plot is closed.
        If False, the program will not be blocked.
        Default value is False.
    similarity_c : float
        The constant to use in the similarity calculation.
        Default value is 1.0.
    """

    sim_c = similarity_c

    labels = utils.one_hot_to_ints(labels)

    n_props = number_of_properties

    # initialize the figure
    fig, axs = plt.subplots(n_props, n_props, figsize=(10,10))

    print(f'Plotting similarity histograms...')

    # loop through all pairs of properties
    cnt = 1
    for i in range(n_props):
        for j in range(n_props):

            # get the features for the current pair of properties
            subset1 = features[labels == i]
            subset2 = features[labels == j]

            # compute the similarities (hard-code with Euclidean and
            # Gaussian for now)
            difs = subset1[:, np.newaxis, :] - subset2[np.newaxis, :, :]
            dsts = np.sqrt(np.sum(difs**2, axis=-1))
            sims = np.exp(-sim_c*dsts**2).ravel()

            # reduce the number of points to improve speed
            if len(sims) > 20000:
                sims = np.random.choice(sims, 20000)
            
            # plot the histogram
            axs[i, j].hist(sims, bins=20)

            # formatting
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
            axs[i, j].set_xlim([0, 1])
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])
            if j == 0:
                axs[i, j].set_ylabel(f'{legend[i]}')
            if i == 0:
                axs[i, j].set_title(f'{legend[j]}')

            print(f'\rFinished {cnt} of {n_props**2}', end='')
            cnt += 1

    if save_path is not None:
        plt.savefig(utils.get_unused_name(save_path), dpi=300)

    if show:    
        plt.show(block=block)

    print('\nDone.')

# ------------------------------------------------------------------------------