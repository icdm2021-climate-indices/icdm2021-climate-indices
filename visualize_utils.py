import os
import pandas as pd
import networkx as nx
import geopandas as gpd

from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon

from evaluation_utils import location_wise_metric


def covariate_visualize(data_file, mask_file, load_directory, timestamp, shape_file=None):
    """
    Function to visualize a given covariate at a given timestamp.

    Arguments:
    - data_file: file name of the data (.h5 format).
    - mask_file: file name of the mask (.h5 format). Specify None to visualize complete data.
    - load_directory: location where `mask_file` and `data_file` are located.
    - timestamp: time stamp in `yyyy-mm-dd` format to view data.
    """
    complete_df = pd.read_hdf(os.path.join(args.load_directory, args.data_file)).reset_index()
    complete_df = complete_df[complete_df.start_date == timestamp]

    if mask_df is not None:
        mask_df = pd.read_hdf(os.path.join(args.load_directory, args.mask_file))
        complete_df = pd.merge(complete_df, mask_df, how="inner", on=["lat", "lon"])

    geometry = [
        Point(lon, lat).buffer(0.25, cap_style=3)
        for lon, lat in zip(complete_df.lon, complete_df.lat)
    ]
    complete_gdf = gpd.GeoDataFrame(complete_df, geometry=geometry)
    fig, ax = plt.subplots()
    complete_gdf.plot(
        column=next(iter(set(complete_df.columns) - set(["lat", "lon", "start_date"]))),
        edgecolor="black",
        linewidth=0.1,
        legend=True,
        ax=ax,
    )
    if shape_file is not None:
        outline_gdf = gpd.read_file(shape_file)
        outline_gdf.plot(color="none", edgecolor="black", ax=ax)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(f"{data_file[:-3]}_{mask_file[:-3]}.pdf", format="pdf")
    plt.close()


def clusters_visualize(graph, save_name, shape_file=None, plot_nino=True, **kwargs):
    """
    Function to visualize clusters from the graph as laid out on a map.

    Arguments:
    - graph: networkx.Graph instance (nodes are clusters) or list of clusters
    - save_name: image save path
    """
    df = pd.DataFrame(columns=["lat", "lon", "cluster_id"])
    if isinstance(graph, nx.Graph):
        coloring = nx.coloring.greedy_color(graph)
        graph_nodes = graph.nodes
        legend = False
    else:
        if kwargs.get("coloring"):
            coloring = kwargs.pop("coloring")
        else:
            coloring = {node: idx for idx, node in enumerate(graph)}
        graph_nodes = graph
        legend = True

    for cluster in graph_nodes:
        if isinstance(cluster[0], tuple):  # Collection of nodes
            for (lat, lon) in cluster:
                df = df.append(
                    {"lat": lat, "lon": lon, "cluster_id": coloring[cluster]}, ignore_index=True
                )
            res_node = min(
                cluster[1:],
                key=lambda node: max(abs(node[0] - cluster[0][0]), abs(node[1] - cluster[0][1])),
            )
            resolution = max(abs(cluster[0][0] - res_node[0]), abs(cluster[0][1] - res_node[1]))
        else:  # Single node
            df = df.append(
                {"lat": cluster[0], "lon": cluster[1], "cluster_id": coloring[cluster]},
                ignore_index=True,
            )
    geometry = [
        Point(lon, lat).buffer(resolution / 2, cap_style=3) for lon, lat in zip(df.lon, df.lat)
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    fig, ax = plt.subplots()
    gdf.plot("cluster_id", edgecolor="none", ax=ax, legend=legend)
    if shape_file is not None:
        outline_gdf = gpd.read_file(shape_file)
        outline_gdf.plot(color="none", edgecolor="black", ax=ax)

    if plot_nino:
        # Plot Nino indices
        geom_nino_1_2 = Polygon([(270, -10), (270, 0), (290, 0), (290, -10)])
        geom_nino_3 = Polygon([(210, -5), (210, 5), (270, 5), (270, -5)])
        geom_nino_4 = Polygon([(180, -5), (180, 5), (210, 5), (210, -5)])
        geom_nino_3_4 = Polygon([(190, -5), (190, 5), (240, 5), (240, -5)])
        gdf_nino = gpd.GeoDataFrame(
            {
                "idx": ["NINO 1+2", "NINO 3", "NINO 4", "NINO 3.4"],
                "geometry": [geom_nino_1_2, geom_nino_3, geom_nino_4, geom_nino_3_4],
            }
        )
        gdf_nino[:3].plot(color="none", edgecolor="black", hatch="/", ax=ax)
        gdf_nino[3:].plot(color="none", edgecolor="black", hatch="\\", ax=ax)

    plt.xlabel("lon")
    plt.ylabel("lat")
    if kwargs.get("plot_title") is not None:
        plot_title = kwargs.pop("plot_title")
        if plot_title != "":
            plt.title(plot_title)
    else:
        plt.title(f"Clusters: {len(graph_nodes)}")
    plt.savefig(f"{save_name}", format=os.path.splitext(save_name)[-1][1:])
    plt.close()


def location_wise_metric_visualize(
    target_data_frame,
    covariate_data_frame,
    std_data_frame,
    graph_or_other,
    save_name,
    metric,
    shape_file=None,
    **kwargs,
):
    total_metric, location_wise_metric_df, cluster_coloring, target_clusters = location_wise_metric(
        target_data_frame=target_data_frame,
        covariate_data_frame=covariate_data_frame,
        std_data_frame=std_data_frame,
        graph_or_other=graph_or_other,
        metric=metric,
        **kwargs,
    )
    res_node = min(
        target_clusters[0][1:],
        key=lambda node: max(
            abs(node[0] - target_clusters[0][0][0]), abs(node[1] - target_clusters[0][0][1])
        ),
    )
    resolution = max(
        abs(target_clusters[0][0][0] - res_node[0]), abs(target_clusters[0][0][1] - res_node[1])
    )
    geometry = [
        Point(lon, lat).buffer(resolution / 2, cap_style=3)
        for lon, lat in zip(location_wise_metric_df.lon, location_wise_metric_df.lat)
    ]
    gdf = gpd.GeoDataFrame(location_wise_metric_df, geometry=geometry)

    fig, ax = plt.subplots()
    gdf.plot("metric", edgecolor="none", ax=ax, legend=True)
    if shape_file is not None:
        outline_gdf = gpd.read_file(shape_file)
        outline_gdf.plot(color="none", edgecolor="black", ax=ax)

    plt.xlabel("lon")
    plt.ylabel("lat")
    if kwargs.get("plot_title") is not None:
        plot_title = kwargs.pop("plot_title")
        if plot_title != "":
            plt.title(plot_title)
    else:
        plt.title(f"Total {metric} : {total_metric:.4f}")
    plt.savefig(
        f"{save_name[:-4]}_location_wise{save_name[-4:]}",
        format=os.path.splitext(save_name)[-1][1:],
    )
    plt.close()

    clusters_visualize(
        target_clusters,
        f"{save_name[:-4]}_cluster_wise{save_name[-4:]}",
        shape_file=shape_file,
        plot_nino=False,
        coloring=cluster_coloring,
        plot_title=f"Average {metric} : {sum(cluster_coloring.values()) / len(cluster_coloring.values()):.4f}",
    )


def weight_matrix_visualize(weight_matrix, save_name, x_groups=None):
    plt.imshow(
        weight_matrix,
        extent=(0, weight_matrix.shape[1], weight_matrix.shape[0], 0),
        cmap="coolwarm",
    )
    if x_groups is not None:
        x_group_width = weight_matrix.shape[1] // x_groups
        xticklabels = []
        for idx, xval in enumerate(range(x_group_width, weight_matrix.shape[1] + 1, x_group_width)):
            plt.plot(
                [xval] * (weight_matrix.shape[0] + 1),
                range(0, weight_matrix.shape[0] + 1),
                "k-",
                linewidth=0.5,
            )
            xticklabels.append(f"C{idx + 1}")
        plt.xticks(range(x_group_width // 2, weight_matrix.shape[1], x_group_width), xticklabels)
    else:
        plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    plt.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    plt.colorbar()
    plt.savefig(save_name, format=os.path.splitext(save_name)[-1][1:])
    plt.close()
