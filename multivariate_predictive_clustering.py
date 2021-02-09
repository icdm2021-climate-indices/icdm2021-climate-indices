import os
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx

from tqdm import tqdm
from argparse import ArgumentParser

from visualize_utils import clusters_visualize
from regression_utils import multiple_regress, r2_score
from multivariate_variance_clustering import (
    construct_graph,
    perform_clustering,
    generate_node_after_contraction,
)


def perform_target_variance_clustering(target_data_frame, num_clusters, silent=False):
    """
    Function to cluster target data into `num_clusters` using the variance based
    agglomerative clustering. Post-clustering, the clustering procedure will extract the mean
    time-series anomalies at each of the `num_clusters` clusters into a DataFrame of size
    `T x num_clusters`, where `T` is the number of timestamps.

    Arguments:
    - target_data_frame: pd.DataFrame object with MultiIndex composed of
                         "lat", "lon" and "start_date" and columns representing the
                         anomalies.
    - num_clusters: number of clusters to be formed.
    - silent: option to mute tqdm
    """
    graph = construct_graph(
        [target_data_frame], edge_weight_assign="variance", validate_graph=True, silent=silent
    )
    graph = perform_clustering(graph, num_clusters, "variance", silent=silent)

    composition_columns = []
    node_list = list(graph.nodes)  # This preserves ordering
    for node in node_list:
        len_node = len(node) if isinstance(node[0], tuple) else 1
        attr = graph.nodes[node]
        composition_columns.append(attr["cov"].reshape(-1, 1) / len_node)
    composition_columns = np.hstack(composition_columns)

    if len_node == 1:
        result_index = target_data_frame[node].index
    else:
        result_index = target_data_frame[node[0]].index
    matrix = pd.DataFrame(composition_columns, index=result_index)
    matrix.index.name = "start_date"
    return matrix, node_list


def create_subsampled_dataset(covariate_data_frames, target_data_frame, date_from, date_to, freq):
    """
    Function to create sub-sampled dataset of the covariates and targets beginning from `date_from`
    till `date_to` every `freq` number of days.

    Arguments:
    - covariate_data_frames: list of pd.DataFrames with MultiIndex composed of "lat", "lon" and "start_date"
                             and columns representing the anomalies.
    - target_data_frame: pd.DataFrame output by the `perform_target_variance_clustering` function with index "start_date".
    - date_from: date in `yyyy-mm-dd` format.
    - date_to: date in `yyyy-mm-dd` format. Should be strictly succeeding `date_from`.
    - freq: integer representing the gap between days
    """
    subsample_indexer = pd.date_range(date_from, date_to, freq=f"{freq}d")
    result_covariate_data_frames = []
    for data_frame in covariate_data_frames:
        tmp = data_frame.unstack(["lat", "lon"]).loc[subsample_indexer]
        tmp.index.name = "start_date"
        tmp = tmp.stack(["lat", "lon"]).reorder_levels(["lat", "lon", "start_date"]).sort_index()
        result_covariate_data_frames.append(tmp)

    subsample_indexer = subsample_indexer.shift(28, freq="d")  # targets are x[t + 28]
    # The covariate dates are d, d + 15, d + 30, d + 45, ....
    # The target dates are d + 28, d + 43, d + 58, d + 73, ....
    # The complete reasoning for this is:
    # - the anomaly at a date d is calculated across the past 14 days.
    # - the covariates are meant to capture this
    # - the targets are supposed to be anomalies computed between days d + 15 to d + 28
    #   which corresponds to the shifted index
    result_target_data_frame = target_data_frame.loc[subsample_indexer]
    result_target_data_frame.index.name = "start_date"
    return result_covariate_data_frames, result_target_data_frame


def multiple_regression_error_weight(graph, node1, node2, reg_param=0):
    """
    Function to compute the r2_score obtained by regressing the combination of covariates
    at two nodes over the general target

    Arguments:
    - graph: networkx.Graph instance
    - node1, node2: nodes composing an edge
    """
    node1_x = graph.nodes[node1]["cov"]
    node2_x = graph.nodes[node2]["cov"]
    len_node1 = len(node1) if isinstance(node1[0], tuple) else 1
    len_node2 = len(node2) if isinstance(node2[0], tuple) else 1
    x_combined = node1_x + node2_x
    w, b = multiple_regress(x_combined, graph.complete_target)
    r2_combined = r2_score(x_combined, graph.complete_target, w, b)
    variance = 0.0
    if reg_param > 0:
        variance = np.mean(np.var(x_combined, axis=0)) / (len_node1 + len_node2) ** 2
    return r2_combined + reg_param * variance


def construct_multivariate_spatial_graph(
    covariate_data_frames, target_data_frame, reg_param=0, silent=False
):
    """
    Function to construct a spatial graph with node attributes set to be the respective sub-sampled anomalies.
    The target is set as a global graph attribute.

    Arguments:
    - covariate_data_frames: list of pd.DataFrames with MultiIndex composed of "lat", "lon" and "start_date"
                             and columns representing the anomalies.
    - target_data_frame: pd.DataFrame output by the `perform_target_variance_clustering` or `create_subsampled_dataset` functions
                         with index "start_date". The lengths of each DataFrame in `covariate_data_frames` and `target_data_frame`
                         should be the same.
    - reg_param: parameter for variance based regularization
    - silent: option to mute tqdm
    """
    # This only creates edges, and no node / edge attributes assignments
    graph = construct_graph([covariate_data_frames[0]], validate_graph=True, silent=silent)
    setattr(graph, "complete_target", target_data_frame.to_numpy(copy=True))

    for data_frame in tqdm(
        covariate_data_frames, ascii=True, desc="meta_node_assign", unit="covariate", disable=silent
    ):
        for node, attr in tqdm(
            data_frame.groupby(level=[0, 1]),
            total=graph.number_of_nodes(),
            desc="node_assign",
            leave=False,
            unit="node",
            ascii=True,
            disable=silent,
        ):

            if graph.nodes[node].get("cov") is None:
                graph.nodes[node]["cov"] = attr.to_numpy(copy=True).reshape(-1, 1)
            else:
                graph.nodes[node]["cov"] = np.hstack(
                    [graph.nodes[node]["cov"], attr.to_numpy(copy=True).reshape(-1, 1)]
                )

    for edge in tqdm(graph.edges, desc="edge_assign", unit="edge", ascii=True, disable=silent):
        graph.edges[edge]["weight"] = multiple_regression_error_weight(
            graph, edge[0], edge[1], reg_param
        )

    return graph


def contract_edge_once(graph, edge, reg_param=0):
    """
    Function to contract a given edge `edge` in the graph `graph`. Covariates of the node resulting from the contraction
    is given by the sum of covariates at the nodes composing `edge`. The edge weight is recalculated after this re-assignment.

    Arguments:
    - graph: networkx.Graph object
    - edge: edge in `graph` to be contracted
    - reg_param: parameter for variance based regularization
    """
    node1, node2 = edge
    new_node = generate_node_after_contraction(node1, node2)

    # Contract nodes, and relabel the contracted node with a tuple indicating the cluster
    new_graph = nx.relabel_nodes(
        nx.contracted_edge(graph, (node1, node2), self_loops=False), {node1: new_node}
    )

    new_graph.nodes[new_node]["cov"] = (
        new_graph.nodes[new_node].pop("cov")
        + new_graph.nodes[new_node].pop("contraction")[node2]["cov"]
    )

    # Re-compute the edge weights
    setattr(new_graph, "complete_target", graph.complete_target)
    for edge in new_graph.edges(new_node):  # Just for the edges which originate at new_node
        new_graph.edges[edge]["weight"] = multiple_regression_error_weight(
            new_graph, edge[0], edge[1], reg_param
        )
    return new_graph


def perform_covariate_predictive_clustering(graph, num_clusters, reg_param=0, silent=False):
    """
    Function to form clusters in the graph by repetitively contracting edges
    with maximum edge weight.

    Arguments:
    - graph: networkx.Graph instance
    - number_of_clusters: number of clusters to be formed. This is equivalent to contracting
                          as many number of times.
    - reg_param: parameter for variance based regularization
    - silent: option to mute tqdm
    """
    num_contractions = (
        graph.number_of_nodes() - num_clusters
    )  # k contractions reduce the number of nodes by k

    # Refreshing the edge weights
    for edge in graph.edges:
        graph.edges[edge]["weight"] = multiple_regression_error_weight(
            graph, edge[0], edge[1], reg_param
        )

    trange = tqdm(
        range(1, num_contractions + 1), desc="contractions", ascii=True, unit="cntr", disable=silent
    )
    for i in trange:
        chosen_edge, max_weight = max(
            nx.get_edge_attributes(graph, "weight").items(), key=lambda edge_weight: edge_weight[1]
        )
        trange.set_postfix({"max_weight": max_weight})
        graph = contract_edge_once(graph, chosen_edge, reg_param)
    return graph


def run_predictive_clustering(
    covariate_data_frames,
    target_data_frame,
    load_directory,
    subsampling_slice,
    num_covariate_clusters,
    num_target_clusters,
    reg_param,
    save_raw_graph,
    save_clustered_graph,
    visualize_formed,
    shape_file,
    silent,
):
    target_data_frame, target_clusters = perform_target_variance_clustering(
        target_data_frame=target_data_frame, num_clusters=num_target_clusters, silent=silent
    )

    date_from, date_to, freq = subsampling_slice.split(":")
    covariate_data_frames, target_data_frame = create_subsampled_dataset(
        covariate_data_frames=covariate_data_frames,
        target_data_frame=target_data_frame,
        date_from=date_from,
        date_to=date_to,
        freq=freq,
    )

    graph = construct_multivariate_spatial_graph(
        covariate_data_frames=covariate_data_frames,
        target_data_frame=target_data_frame,
        reg_param=reg_param,
        silent=silent,
    )
    if save_raw_graph is not None:
        pkl.dump(graph, open(os.path.join(load_directory, save_raw_graph + ".pkl"), "wb"))

    graph = perform_covariate_predictive_clustering(
        graph=graph, num_clusters=num_covariate_clusters, reg_param=reg_param, silent=silent
    )
    setattr(graph, "target_clusters", target_clusters)
    if save_clustered_graph is not None:
        pkl.dump(graph, open(os.path.join(load_directory, save_clustered_graph + ".pkl"), "wb"))

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
        "--covariate_data_files",
        required=True,
        nargs="+",
        type=str,
        help="Specify the anomaly data files (in .h5 format)",
    )
    p.add_argument(
        "--target_data_file",
        required=True,
        type=str,
        help="Specify the target data file (in .h5 format)",
    )
    p.add_argument(
        "--load_directory", required=True, type=str, help="Specify the loading directory"
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
    p.add_argument(
        "--num_covariate_clusters",
        type=int,
        required=True,
        help="Specify the number of clusters (covariate clustering)",
    )
    p.add_argument(
        "--num_target_clusters",
        type=int,
        required=True,
        help="Specify the number of clusters (target clustering)",
    )
    p.add_argument(
        "--subsampling_slice",
        type=str,
        default="1981-09-26:2010-12-03:15",
        help="Specify the slice (in begin_date:end_date:freq format)",
    )
    p.add_argument(
        "--visualize_formed",
        default=None,
        type=str,
        help="Specify file name to plot the formed clusters",
    )
    p.add_argument(
        "--shape_file", default=None, type=str, help="Specify shape file for land borders"
    )
    p.add_argument("--silent", action="store_true", help="Toggle to mute progress bar")
    p.add_argument(
        "--reg_param", default=0, type=float, help="Variance based regularization parameter"
    )
    p.add_argument(
        "--preprocess_date_range",
        default=None,
        type=str,
        help="Specify date range to consider (in begin_date:end_date format)",
    )
    args = p.parse_args()

    covariate_data_frames = []
    for data_file in args.covariate_data_files:
        df = pd.read_hdf(os.path.join(args.load_directory, data_file))
        if args.preprocess_date_range is not None:
            date_from, date_to = args.preprocess_date_range.split(":")
            df = df.unstack(["lat", "lon"])
            df = df.loc[pd.date_range(date_from, date_to)]
            df.index.name = "start_date"
            df = df.stack(["lat", "lon"]).reorder_levels(["lat", "lon", "start_date"]).sort_index()
        covariate_data_frames.append(df)

    target_data_frame = pd.read_hdf(os.path.join(args.load_directory, args.target_data_file))
    if args.preprocess_date_range is not None:
        date_from, date_to = args.preprocess_date_range.split(":")
        target_data_frame = target_data_frame.unstack(["lat", "lon"])
        target_data_frame = target_data_frame.loc[pd.date_range(date_from, date_to)]
        target_data_frame.index.name = "start_date"
        target_data_frame = (
            target_data_frame.stack(["lat", "lon"])
            .reorder_levels(["lat", "lon", "start_date"])
            .sort_index()
        )

    run_predictive_clustering(
        covariate_data_frames=covariate_data_frames,
        target_data_frame=target_data_frame,
        load_directory=args.load_directory,
        subsampling_slice=args.subsampling_slice,
        num_covariate_clusters=args.num_covariate_clusters,
        num_target_clusters=args.num_target_clusters,
        reg_param=args.reg_param,
        save_raw_graph=args.save_raw_graph,
        save_clustered_graph=args.save_clustered_graph,
        visualize_formed=args.visualize_formed,
        shape_file=args.shape_file,
        silent=args.silent,
    )
