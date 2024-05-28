"""
this code has been adapted from the napari-arboretum on github (url: https://github.com/lowe-lab-ucl/arboretum/blob/main/src/napari_arboretum), and modified to work without napari visualisation
"""

# Import necessary libraries
import networkx as nx
import btrack
from btrack import utils, config
import matplotlib.pyplot as plt
from traccuracy.loaders import load_ctc_data
import napari
import ctctools
import numpy as np
import os
import textwrap
from napari_arboretum.tree import *
from napari_arboretum.graph import *
from optimization import compute_scaling_factors

import matplotlib as mpl

# Setting font to Helvetica and increasing the font size globally
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16



class HashableTreeNode:
    def __init__(self, node):
        self.node = node

    def __hash__(self):
        return hash(self.node.ID)

    def __eq__(self, other):
        return self.node.ID == other.node.ID

    def __str__(self):
        return f"HashableTreeNode(id={self.node.ID}, label={self.node.label})"

def set_generation_attributes(G):
    """
    Set the generation attributes for each node in the graph.

    Parameters:
    G : networkx.Graph
        The graph whose nodes' generation attributes are to be set.
    """
    for node in G.nodes():
        G.nodes[node]['generation'] = count_parents(G, node)

def draw_graph(G):
    """
    Draw the graph with its subgraphs.

    Parameters:
    G : networkx.Graph
        The graph to be drawn.
    """
    print("Drawing graph...")
    # Find the unique track IDs
    unique_track_ids = np.unique([node.split('_')[0] for node in G.nodes()])

    # Create a new figure for the combined plot
    fig, ax = plt.subplots()

    # Initialize the y-offset
    y_offset = 0

    # Create a set to store nodes of plotted subgraphs
    plotted_subgraphs = set()

    # Create a plot for each unique track ID
    for search_node in unique_track_ids:
        # Build the subgraph
        nodes = build_subgraph(G, search_node)
        print(nodes)

        # Convert nodes to a frozenset so it can be added to a set
        nodes_set = frozenset(HashableTreeNode(node) for node in nodes)

        # Check if this subgraph's nodes are already in the set of plotted subgraphs
        if nodes_set in plotted_subgraphs:
            continue  # Skip this iteration if the subgraph has already been plotted

        # Add this subgraph's nodes to the set of plotted subgraphs
        plotted_subgraphs.add(nodes_set)

        # Layout the tree
        edges, annotations = layout_tree(nodes)

        # Adjust the y-coordinates of the edges and annotations by the y-offset
        for edge in edges:
            edge.y = tuple(y + y_offset for y in edge.y)
        for annotation in annotations:
            annotation.y += y_offset

        # Plot edges
        for edge in edges:
            ax.plot(edge.x, edge.y, color='blue', linewidth=3)

        # Add annotations
        for annotation in annotations:
            ax.text(annotation.x, annotation.y, annotation.label, color='red')

        # Update the y-offset for the next track
        y_offset += 4

    # Show the plot
    plt.show()

def load_from_dict(dict):
    """
    Load the configuration from a dictionary.

    Parameters:
    dict : dict
        The dictionary containing configuration parameters.

    Returns:
    btrack.config.Config
        The loaded configuration object.
    """
    conf = config.load_config('cell_config.json')  # config
    attributes = {
        'theta_dist': dict['theta_dist'],
        'lambda_time': dict['lambda_time'],
        'lambda_dist': dict['lambda_dist'],
        'lambda_link': dict['lambda_link'],
        'lambda_branch': dict['lambda_branch'],
        'theta_time': dict['theta_time'],
        'dist_thresh': dict['dist_thresh'],
        'time_thresh': dict['time_thresh'],
        'apop_thresh': dict['apop_thresh'],
        'segmentation_miss_rate': dict['segmentation_miss_rate'],
        'P': scale_matrix(conf.motion_model.P, 150.0, dict['p_sigma']),
        'G': scale_matrix(conf.motion_model.G, 15.0, dict['g_sigma']),
        'R': scale_matrix(conf.motion_model.R, 5.0, dict['r_sigma']),
        'accuracy': dict['accuracy'],
        'max_lost': dict['max_lost'],
        'prob_not_assign': dict['prob_not_assign']
    }

    # Set attributes for hypothesis model and motion model
    for attr, value in attributes.items():
        if attr in ['P', 'G', 'R', 'max_lost', 'prob_not_assign', 'accuracy']:
            setattr(conf.motion_model, attr, value)
        else:
            setattr(conf.hypothesis_model, attr, value)

    # Set division hypothesis
    if dict['div_hypothesis'] == 1:
        setattr(conf.hypothesis_model, 'hypotheses', [
            "P_FP",
            "P_init",
            "P_term",
            "P_link",
            "P_branch",
            "P_dead"
        ])
    elif dict['div_hypothesis'] == 0:
        setattr(conf.hypothesis_model, 'hypotheses', [
            "P_FP",
            "P_init",
            "P_term",
            "P_link",
            "P_dead"
        ])
    else:
        raise ValueError(f"Invalid value for div_hypothesis: {dict['div_hypothesis']}. It should be 0 or 1.")
    return conf

def scale_matrix(matrix: np.ndarray, original_sigma: float, new_sigma: float) -> np.ndarray:
    """
    Scale a matrix by first reverting the original scaling and then applying a new sigma value.

    Parameters:
    matrix : np.ndarray
        The matrix to be scaled.
    original_sigma : float
        The original sigma value used to scale the matrix.
    new_sigma : float
        The new sigma value to scale the matrix.

    Returns:
    np.ndarray
        The rescaled matrix.
    """
    # Revert the original scaling
    if original_sigma != 0:
        unscaled_matrix = matrix / original_sigma
    else:
        unscaled_matrix = matrix.copy()  # Avoid division by zero

    # Apply the new sigma scaling
    rescaled_matrix = unscaled_matrix * new_sigma

    return rescaled_matrix

def initialize_config(dict=None, config_file_path='cell_config.json', load_from_dict_flag=True):
    """
    Initialize the configuration either from a dictionary or a JSON file.

    Parameters:
    dict : dict, optional
        The dictionary containing configuration parameters. Default is None.
    config_file_path : str, optional
        The path to the JSON configuration file. Default is 'cell_config.json'.
    load_from_dict_flag : bool, optional
        Flag indicating whether to load from dictionary or JSON file. Default is True.

    Returns:
    btrack.config.Config
        The initialized configuration object.
    """
    if load_from_dict_flag and dict is not None:
        conf = load_from_dict(dict)
    else:
        conf = config.load_config(config_file_path)
    
    return conf

def load_and_configure_tracker(data_path, conf, scale):
    """
    Load and configure the tracker with the given data and configuration.

    Parameters:
    data_path : str
        The path to the data.
    conf : btrack.config.Config
        The configuration object.
    scale : tuple
        The scaling factors.

    Returns:
    tuple
        The tracked data, properties, graph, and dataset.
    """
    # Load the dataset
    dataset = ctctools.load(data_path, experiment="01", scale=scale)
    objects = utils.segmentation_to_objects(dataset.segmentation, properties=('area', ))
    volume = dataset.volume

    with btrack.BayesianTracker(verbose=True) as tracker:
        tracker.configure(conf)
        tracker.append(objects)
        tracker.volume = volume[::-1]  # need to reverse it to fit btrack requirements
        tracker.track(step_size=100)
        tracker.optimize()
        data, properties, graph = tracker.to_napari()
        
    return data, properties, graph, dataset

def plot_lineage_tree(data, properties, graph, dataset, data_name, output_path):
    """
    Plot the lineage tree.

    Parameters:
    data : np.ndarray
        The tracked data.
    properties : dict
        The properties of the tracked data.
    graph : networkx.Graph
        The graph of the tracked data.
    dataset : object
        The dataset object.
    data_name : str
        The name of the data.
    output_path : str
        The path to save the output image.
    """
    viewer = napari.Viewer(show=False)
    viewer.add_labels(dataset.segmentation)
    layer = viewer.add_tracks(data, properties=properties, graph=graph)

    # Find the unique track IDs
    unique_track_ids = np.unique(layer.properties['track_id'])

    # Create a new figure for the combined plot
    fig, ax = plt.subplots(figsize=(8, 16), dpi=300)

    # Add title
    title = 'Lineage Tree Predicted by btrack'
    plt.title("\n".join(textwrap.wrap(title, 35)), loc='left')

    # Add x-axis label
    plt.xlabel('Time (frames)')

    # Remove y-axis ticks and labels
    plt.yticks([])

    # Initialize the y-offset
    y_offset = 0

    # Create a set to store nodes of plotted subgraphs
    plotted_subgraphs = set()

    # Create a plot for each unique track ID
    for search_node in unique_track_ids:
        # Build the subgraph
        nodes = build_subgraph(layer, search_node)

        # Convert nodes to a frozenset so it can be added to a set
        nodes_set = frozenset(HashableTreeNode(node) for node in nodes)

        # Check if this subgraph's nodes are already in the set of plotted subgraphs
        if nodes_set in plotted_subgraphs:
            continue  # Skip this iteration if the subgraph has already been plotted

        # Add this subgraph's nodes to the set of plotted subgraphs
        plotted_subgraphs.add(nodes_set)

        # Layout the tree
        edges, annotations = layout_tree(nodes)

        # Adjust the y-coordinates of the edges and annotations by the y-offset
        for edge in edges:
            edge.y = tuple(y + y_offset for y in edge.y)
        for annotation in annotations:
            annotation.y += y_offset

        # Plot edges
        for edge in edges:
            ax.plot(edge.x, edge.y, color='blue', linewidth=3)

        # Update the y-offset for the next track
        y_offset += 4

    plt.savefig(os.path.join(output_path, f'{data_name}.png'), dpi=300)

if __name__ == "__main__":
    datasets_path = "downloads" # Directory where the dataset is stored
    data_name = 'Fluo-N2DL-HeLa' # Specify the name of the dataset
    data_path = os.path.join(datasets_path, data_name)
    output_path = "pred_trees"  # Specify the output path for the image

    dict = {
        "theta_dist": 47.823413444235534,
        "lambda_time": 21.254843580963723,
        "lambda_dist": 8.128005345050447,
        "lambda_link": 52.50747518465564,
        "lambda_branch": 86.70087091013778,
        "theta_time": 0.02247196274546187,
        "dist_thresh": 58.87030503535955,
        "time_thresh": 1.9295917636949447,
        "apop_thresh": 5,
        "segmentation_miss_rate": 0.2246097893614105,
        "p_sigma": 9.667962324738056,
        "g_sigma": 20.822081259646865,
        "r_sigma": 207.01278244282472,
        "accuracy": 6.349091082377734,
        "max_lost": 9,
        "prob_not_assign": 0.22731137131927692,
        "max_search_radius": 509,
        "div_hypothesis": 1
    } # Specify the configuration parameters

    voxel_sizes = (0.645, 0.645, 1.0)  # Voxel size in microns, write third value as 1.0 if data is 2D
    scale = compute_scaling_factors(voxel_sizes)  # Scale can be specified here
    load_from_dict_flag = True  # Specify if the config should be loaded from dict or JSON file

    conf = initialize_config(dict=dict, load_from_dict_flag=load_from_dict_flag)
    data, properties, graph, dataset = load_and_configure_tracker(data_path, conf, scale)
    plot_lineage_tree(data, properties, graph, dataset, data_name, output_path)