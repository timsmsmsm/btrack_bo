"""
This code has been adapted from the napari-arboretum on GitHub (URL: https://github.com/lowe-lab-ucl/arboretum),
and modified to work without napari visualization.
"""

# Import necessary libraries
from __future__ import annotations
from dataclasses import dataclass, field

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from traccuracy.loaders import load_ctc_data
from traccuracy.loaders._ctc import ctc_to_graph, _get_node_attributes
import numpy as np
import os
from collections import deque
from napari_arboretum.tree import *
from napari_arboretum.graph import *
import matplotlib as mpl

# Setting font to Helvetica and increasing the font size globally
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

def load_data_and_create_graph(data_path):
    """
    Load data from the specified path and create a directed graph.

    Parameters:
    data_path : str
        The path to the dataset.

    Returns:
    networkx.DiGraph
        The created directed graph with edges from the ground truth data.
    """
    gt_data = load_ctc_data(data_path + '/01_GT/TRA', data_path + '/01_GT/TRA/man_track.txt')
    gt_G = nx.DiGraph()
    gt_G.add_edges_from(gt_data.edges())
    return gt_G

def convert_gt_G_to_layer_format(gt_G):
    """
    Convert the ground truth graph to a format suitable for visualization layers.

    Parameters:
    gt_G : networkx.DiGraph
        The ground truth graph.

    Returns:
    tuple
        A tuple containing the data array and properties dictionary.
    """
    data = []

    for node in gt_G.nodes():
        cell, timestep = node.split('_')
        cell = int(cell)
        timestep = int(timestep)
        row = [cell, timestep, 0, 0, 0]
        data.append(row)

    data = np.array(data)
    properties = {}
    return data, properties

def convert_graph_to_dict(graph):
    """
    Convert a graph to a dictionary format.

    Parameters:
    graph : networkx.DiGraph
        The graph to convert.

    Returns:
    dict
        A dictionary where keys are nodes and values are lists of predecessors.
    """
    graph_dict = {}
    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))
        graph_dict[node] = predecessors

    for node in graph.nodes():
        if node not in graph_dict:
            graph_dict[node] = []

    return graph_dict

def convert_dict_to_edge_format(graph_dict):
    """
    Convert a graph dictionary to edge format.

    Parameters:
    graph_dict : dict
        The graph dictionary to convert.

    Returns:
    dict
        A dictionary where keys are edge IDs and values are source-target pairs.
    """
    edge_dict = {}
    edge_id = 0
    for target, sources in graph_dict.items():
        for source in sources:
            edge_dict[edge_id] = {'source': source, 'target': target}
            edge_id += 1
    return edge_dict

def layout_tree_2(G):
    """
    Layout a tree for plotting.

    Parameters:
    G : networkx.DiGraph
        The tree to layout.

    Returns:
    tuple
        A tuple containing the list of edges and annotations for plotting.
    """
    if not nx.is_tree(G):
        raise ValueError("Input graph must be a tree")

    root = [n for n, d in G.in_degree() if d == 0][0]

    queue = [root]
    marked = [root]
    y_pos = {root: 0.0}

    edges: list[Edge] = []
    annotations: list[Annotation] = []

    while queue:
        node = queue.pop(0)
        y = y_pos[node]

        edges.append(
            Edge(y=(y, y), x=(G.nodes[node]['t'][0], G.nodes[node]['t'][-1]), track_id=node, node=node)
        )

        children = list(G.successors(node))

        depth_mod = 2.0 / (2.0 ** (G.nodes[node]['generation']))
        spacing = np.linspace(-depth_mod, depth_mod, len(children))
        y_mod = spacing if len(children) > 1 else np.array([0.0])

        for idx, child in enumerate(children):
            if child not in marked:
                marked.append(child)
                queue.append(child)
                y_pos[child] = y + y_mod[idx]

    for node in G.nodes():
        if node not in marked:
            marked.append(node)
            queue.append(node)
            y_pos[node] = y + 1  # Or some other logic to determine y position

            edges.append(
                Edge(y=(y, y_pos[node]), x=(G.nodes[node]['t'][0], G.nodes[node]['t'][-1]), track_id=node, node=node)
            )

    for node in G.nodes:
        if G.out_degree(node) > 0:
            children = list(G.successors(node))
            for child in children:
                edges.append(
                    Edge(
                        y=(y_pos[node], y_pos[child]),
                        x=(G.nodes[node]['t'][-1], G.nodes[child]['t'][0]),
                    )
                )

    return edges, annotations

def count_parents(G, node):
    """
    Count the number of parent nodes for a given node in the graph.

    Parameters:
    G : networkx.DiGraph
        The graph containing the nodes.
    node : Any
        The node for which to count parents.

    Returns:
    int
        The number of parent nodes.
    """
    count = 0
    while True:
        parents = list(G.predecessors(node))
        if not parents:
            break
        node = parents[0]
        if len(list(G.successors(node))) >= 2:
            count += 1
    return count

def process_and_plot_ground_truth(data_path, output_path, data_name):
    """
    Process the ground truth data and plot the lineage tree.

    Parameters:
    data_path : str
        The path to the dataset.
    output_path : str
        The path to save the output image.
    data_name : str
        The name of the dataset.
    """
    path = os.path.join(data_path, data_name)
    if os.path.isdir(path):
        gt_G = load_data_and_create_graph(path)
        for node in gt_G.nodes():
            cell, timestep = node.split('_')
            timestep = int(timestep)
            gt_G.nodes[node]['t'] = [timestep, timestep]

        fig, ax = plt.subplots(figsize=(8, 16))
        plt.title('Ground Truth Lineage Tree', loc='left')
        plt.xlabel('Time (frames)')
        plt.yticks([])

        y_offset = 0
        components = list(nx.weakly_connected_components(gt_G))
        components.sort(key=len)

        for i, tree_nodes in enumerate(components):
            tree = gt_G.subgraph(tree_nodes)
            root = [n for n, d in tree.in_degree() if d == 0][0]

            nx.set_node_attributes(tree, {node: count_parents(tree, node) for node in tree.nodes()}, 'generation')
            edges, annotations = layout_tree_2(tree)

            for edge in edges:
                edge.y = tuple(y + y_offset for y in edge.y)
            for annotation in annotations:
                annotation.y += y_offset

            for edge in edges:
                ax.plot(edge.x, edge.y, color='blue', linewidth=3)

            for annotation in annotations:
                ax.text(annotation.x, annotation.y, annotation.label, color='red')

            Y_INCREMENT = 8
            y_offset += Y_INCREMENT

        max_generation = max(nx.get_node_attributes(tree, 'generation').values())
        print(f"The largest generation in the tree is: {max_generation}")

        ax.set_ylim(-Y_INCREMENT, y_offset)
        dpi = 300

        plt.savefig(os.path.join(output_path, f"{data_name}.png"), dpi=dpi)

if __name__ == "__main__":
    datasets_path = "downloads" # Directory where the dataset is stored
    data_name = 'Fluo-N2DL-HeLa' # Name of the dataset
    data_path = os.path.join(datasets_path, data_name)
    output_path = "gt_trees" # Directory to save the output image

    process_and_plot_ground_truth(datasets_path, output_path, data_name)
