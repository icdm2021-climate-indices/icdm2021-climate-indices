import os
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.spatial.distance as ssd

from tqdm import tqdm
from argparse import ArgumentParser

from visualize_utils import clusters_visualize


def obtain_vertices(data_frame):
    """
    Function to extract the vertices of the graph from `data_frame` by manipulating the index.

    Arguments:
    - data_frame: pd.DataFrame object with MultiIndex index composed of
                  "lat", "lon" and "start_date"
    """
    vertex_list = data_frame.index.droplevel("start_date").unique()
    return vertex_list.tolist()


def variance_edge_weight(graph, node1, node2):
    """
    Function to compute the combined normalized variance between two nodes in `graph`.

    Given two sets of observations, the combined normalized variance is given by
    empirical variance of X1 union X2 divided by the square of sum of sizes.

    Arguments:
    - graph: networkx.Graph object with node metadata containing anomalies.
    - node1, node2: nodes to compute combined normalized variance.
    """
    node1_values = graph.nodes[node1]["cov"]
    node2_values = graph.nodes[node2]["cov"]
    len_node1 = len(node1) if isinstance(node1[0], tuple) else 1
    len_node2 = len(node2) if isinstance(node2[0], tuple) else 1
    combined_var = np.mean(np.var(node1_values + node2_values, axis=0))
    return combined_var / (len_node1 + len_node2) ** 2


def get_edge_weight_func(edge_weight_assign):
    """
    Function to return a callable based on edge_weight_assign.

    Arguments:
    - edge_weight_assign: string representing the weighting function.
    """
    if edge_weight_assign == "variance":
        return variance_edge_weight
    else:
        raise NotImplementedError("Other weighting functions not ready")


def construct_graph(data_frames, edge_weight_assign=None, validate_graph=False, silent=False):
    """
    Function construct a graph object given a data_frame and an edge weight assignment function.
    Optionally, validate the graphby checking spatial contiguity.

    Arguments:
    - data_frames: list of pd.DataFrame objects, each with MultiIndex index composed of
                  "lat", "lon" and "start_date" and columns representing the anomalies.
    - edge_weight_assign: string representing the weighting function. (default: None)
    - validate_graph: option to validate spatial contiguity between nodes in the graph
                      (default: False)
    - silent: option to silence tqdm progress bar
    """
    combined_df = data_frames[0].reset_index()
    for data_frame in data_frames[1:]:
        combined_df = pd.merge(
            combined_df, data_frame.reset_index(), on=["lat", "lon", "start_date"], how="inner"
        )
    combined_df = combined_df.set_index(["lat", "lon", "start_date"]).sort_index()
    vertices = obtain_vertices(data_frames[0])
    resolution = min(ssd.pdist(vertices))
    adjacency_matrix = pd.DataFrame(
        ssd.squareform(ssd.pdist(vertices)) == resolution, columns=vertices, index=vertices
    )  # This will create the nodes and edges
    graph = nx.from_pandas_adjacency(adjacency_matrix)
    graph = nx.OrderedGraph(graph)

    if edge_weight_assign is not None:
        edge_weight_func = get_edge_weight_func(edge_weight_assign)
        # Add the node attributes
        for node, attr in tqdm(
            combined_df.groupby(level=[0, 1]),
            total=graph.number_of_nodes(),
            desc="node_assign",
            unit="node",
            ascii=True,
            disable=silent,
        ):
            graph.nodes[node]["cov"] = attr.to_numpy(copy=True)

        for edge in tqdm(graph.edges, desc="edge_assign", unit="edge", ascii=True, disable=silent):
            graph.edges[edge]["weight"] = edge_weight_func(
                graph, edge[0], edge[1]
            )  # Add the edge attributes

    if validate_graph:
        for edge in tqdm(
            graph.edges, desc="validate_graph", unit="edge", ascii=True, disable=silent
        ):
            assert (
                (abs(edge[0][0] - edge[1][0]) == resolution) and (edge[0][1] == edge[1][1])
            ) or (
                (abs(edge[0][1] - edge[1][1]) == resolution) and (edge[0][0] == edge[1][0])
            ), f"Graph creation failed, {edge} is incorrectly added"
    return graph


def generate_node_after_contraction(node1, node2):
    if isinstance(node1[0], tuple):
        if isinstance(node2[0], tuple):
            # This case is when node1 and node2 are tuples of coordinates
            new_node = node1 + node2
        else:
            # This case is when node1 is a tuple of coordinates and node2 is a single coordinate
            new_node = node1 + (node2,)
    else:
        if isinstance(node2[0], tuple):
            # This case is when node1 is a single coordinate and node2 is a tuple of coordinates
            new_node = (node1,) + node2
        else:
            # This case is when node1 and node2 are single coordinates
            new_node = (node1,) + (node2,)
    return new_node


def contract_edge(graph, edge, edge_weight_assign):
    """
    Function to contract an edge in the graph. Attributes of the nodes of edge being contracted
    are appended to form one longer array and is assigned to the new node resulting from the
    contraction. The edge weight is recalculated after this re-assignment.

    Arguments:
    - graph: networkx.Graph object
    - edge: edge in `graph` to be contracted
    - edge_weight_assign: string representing the weighting function.
    """
    node1, node2 = edge
    new_node = generate_node_after_contraction(node1, node2)

    # Contract nodes, and relabel the contracted node with a tuple indicating the cluster
    new_graph = nx.relabel_nodes(
        nx.contracted_edge(graph, (node1, node2), self_loops=False), {node1: new_node}
    )

    # Re-assign the anomaly attribute; this is simply a concatenation of existing anomalies
    new_graph.nodes[new_node]["cov"] = (
        new_graph.nodes[new_node].pop("cov")
        + new_graph.nodes[new_node].pop("contraction")[node2]["cov"]
    )

    # Re-compute the edge weights
    edge_weight_func = get_edge_weight_func(edge_weight_assign)
    for edge in new_graph.edges(new_node):
        new_graph.edges[edge]["weight"] = edge_weight_func(new_graph, edge[0], edge[1])
    return new_graph


def perform_clustering(graph, number_of_clusters, edge_weight_assign, silent=False):
    """
    Function to form clusters in the graph by repetitively contracting edges
    with maximum edge weight.

    Arguments:
    - graph: networkx.Graph instance
    - number_of_clusters: number of clusters to be formed. This is equivalent to contracting
                          as many number of times.
    - edge_weight_assign: string representing the weighting function.
    - silent: option to silence tqdm progress bar (default: False)
    """
    num_contractions = (
        graph.number_of_nodes() - number_of_clusters
    )  # k contractions reduce the number of nodes by k

    # Refreshing the edge weights
    edge_weight_func = get_edge_weight_func(edge_weight_assign)
    for edge in graph.edges:
        graph.edges[edge]["weight"] = edge_weight_func(graph, edge[0], edge[1])

    trange = tqdm(
        range(1, num_contractions + 1), desc="contractions", ascii=True, unit="cntr", disable=silent
    )
    for i in trange:
        chosen_edge, max_weight = max(
            nx.get_edge_attributes(graph, "weight").items(), key=lambda edge_weight: edge_weight[1]
        )
        trange.set_postfix({"max_weight": max_weight})
        graph = contract_edge(graph, chosen_edge, edge_weight_assign)
    return graph


def run_variance_clustering(
    data_frames,
    load_directory,
    num_clusters,
    save_raw_graph,
    save_clustered_graph,
    visualize_formed,
    shape_file,
    silent,
):
    graph = construct_graph(
        data_frames=data_frames, edge_weight_assign="variance", validate_graph=True, silent=silent
    )
    if save_raw_graph is not None:
        pkl.dump(graph, open(os.path.join(load_directory, f"{save_raw_graph}.pkl"), "wb"))

    graph = perform_clustering(
        graph=graph, number_of_clusters=num_clusters, edge_weight_assign="variance", silent=silent
    )
    if save_clustered_graph:
        pkl.dump(graph, open(os.path.join(load_directory, f"{save_clustered_graph}.pkl"), "wb"))

    if visualize_formed is not None:
        clusters_visualize(
            graph,
            os.path.join(load_directory, visualize_formed + ".pdf"),
            shape_file,
            plot_nino=False,
        )
    return graph


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--data_files",
        nargs="+",
        type=str,
        required=True,
        help="Specify data file names (in .h5 format)",
    )
    p.add_argument(
        "--load_directory",
        type=str,
        required=True,
        help="Specify the location of data_file or graph_file",
    )
    p.add_argument(
        "--save_raw_graph",
        default=None,
        type=str,
        help="If specified, save created graph in load_directory with argument as file name",
    )
    p.add_argument(
        "--save_clustered_graph",
        default=None,
        type=str,
        help="If specified, save clustered graph in load_directory with argument as file name",
    )
    p.add_argument("--num_clusters", type=int, required=True, help="Specify the number of clusters")
    p.add_argument(
        "--visualize_formed",
        default=None,
        type=str,
        help="Specify file name to plot the formed clusters",
    )
    p.add_argument(
        "--shape_file", default=None, type=str, help="Specify shape file for land borders"
    )
    p.add_argument(
        "--preprocess_date_range",
        default=None,
        type=str,
        help="Specify date range to consider (in begin_date:end_date format)",
    )
    p.add_argument("--silent", action="store_true", help="Toggle to mute progress bar")
    args = p.parse_args()

    data_frames = []
    for data_file in args.data_files:
        df = pd.read_hdf(os.path.join(args.load_directory, data_file))
        if args.preprocess_date_range is not None:
            date_from, date_to = args.preprocess_date_range.split(":")
            df = df.unstack(["lat", "lon"])
            df = df.loc[pd.date_range(date_from, date_to)]
            df.index.name = "start_date"
            df = df.stack(["lat", "lon"]).reorder_levels(["lat", "lon", "start_date"]).sort_index()

        data_frames.append(df)

    run_variance_clustering(
        data_frames=data_frames,
        load_directory=args.load_directory,
        num_clusters=args.num_clusters,
        save_raw_graph=args.save_raw_graph,
        save_clustered_graph=args.save_clustered_graph,
        visualize_formed=args.visualize_formed,
        shape_file=args.shape_file,
        silent=args.silent,
    )
