import os
import pandas as pd

from itertools import product
from argparse import ArgumentParser

from evaluation_utils import location_wise_metric
from multivariate_variance_clustering import run_variance_clustering
from multivariate_predictive_clustering import (
    run_predictive_clustering,
    perform_target_variance_clustering,
)


def return_date_range_max_min(data_frame, min_date=None, max_date=None):
    tmp = data_frame.unstack(["lat", "lon"])
    tmp_index = tmp.index
    if min_date is not None:
        tmp = tmp.loc[tmp_index[tmp_index >= min_date]]
    elif max_date is not None:
        tmp = tmp.loc[tmp_index[tmp_index < max_date]]
    else:
        raise ValueError("min_date and max_date cannot be None simultaneously")
    tmp.index.name = "start_date"
    tmp = tmp.stack(["lat", "lon"]).reorder_levels(["lat", "lon", "start_date"]).sort_index()
    return tmp


def return_date_range_slice(data_frame, date_range):
    tmp = data_frame.unstack(["lat", "lon"])
    tmp = tmp.loc[date_range]
    tmp.index.name = "start_date"
    tmp = tmp.stack(["lat", "lon"]).reorder_levels(["lat", "lon", "start_date"]).sort_index()
    return tmp


def construct_required_data(
    covariate_data_frames,
    target_data_frame,
    target_data_frame_unstandardized,
    subsampling_slice,
    begin_valid_year,
):
    # For the new training set, we will be considering the slice from 1981 - begin_valid_year.
    # This will just be a direct slice, and no aggregation is performed pertaining to
    # these years separately.
    # It is important to note that training data statistics such as day-wise std
    # is computed on the entire training set (1981 - 2010).
    # Any subsequent rescaling is performed with this std.
    new_train_covariate_data_frames = []
    new_valid_covariate_data_frames = []
    for cov_df in covariate_data_frames:
        tmp_train = return_date_range_max_min(cov_df, max_date=f"{begin_valid_year}-01-01")
        new_train_covariate_data_frames.append(tmp_train)
        tmp_valid = return_date_range_max_min(cov_df, min_date=f"{begin_valid_year}-01-01")
        new_valid_covariate_data_frames.append(tmp_valid)

    new_train_target_data_frame = return_date_range_max_min(
        target_data_frame, max_date=f"{begin_valid_year}-01-01"
    )
    new_valid_target_data_frame = return_date_range_max_min(
        target_data_frame_unstandardized, min_date=f"{begin_valid_year}-01-01"
    )

    date_from, date_to, freq = subsampling_slice.split(":")
    subsample_indexer = pd.date_range(date_from, date_to, freq=f"{freq}d")
    new_train_indexer = subsample_indexer[subsample_indexer < f"{begin_valid_year - 1}-12-04"]
    # the significance of that specific date is that, anything further than 12 04,
    # would require a target in the validation range
    new_train_date_from, new_train_date_to = (
        new_train_indexer[0].strftime("%Y-%m-%d"),
        new_train_indexer[-1].strftime("%Y-%m-%d"),
    )
    new_train_subsampling_slice = f"{new_train_date_from}:{new_train_date_to}:{freq}"

    valid_date_range = pd.date_range(f"{begin_valid_year}-01-14", "2010-12-03", freq=f"{freq}d")
    new_valid_covariate_data_frames = pd.concat(
        [
            return_date_range_slice(cov_df, valid_date_range)
            for cov_df in new_valid_covariate_data_frames
        ],
        axis=1,
    )
    new_valid_target_data_frame = return_date_range_slice(
        new_valid_target_data_frame, valid_date_range.shift(28, freq="d")
    )

    return (
        new_train_covariate_data_frames,
        new_train_target_data_frame,
        new_valid_covariate_data_frames,
        new_valid_target_data_frame,
        new_train_subsampling_slice,
    )


def run_one_configuration(
    new_train_covariate_data_frames,
    new_train_target_data_frame,
    new_valid_covariate_data_frames,
    new_valid_target_data_frame,
    new_train_subsampling_slice,
    std_data_frame,
    num_covariate_clusters,
    num_target_clusters,
    reg_param,
    clustering_algo,
    log_file,
):
    if clustering_algo == "predictive":
        graph = run_predictive_clustering(
            covariate_data_frames=new_train_covariate_data_frames,
            target_data_frame=new_train_target_data_frame,
            load_directory=None,
            subsampling_slice=new_train_subsampling_slice,
            num_covariate_clusters=num_covariate_clusters,
            num_target_clusters=num_target_clusters,
            reg_param=reg_param,
            save_raw_graph=None,
            save_clustered_graph=None,
            visualize_formed=None,
            shape_file=None,
            silent=True,
        )

    elif clustering_algo == "variance":
        # Perform covariate variance clustering
        graph = run_variance_clustering(
            data_frames=new_train_covariate_data_frames,
            load_directory=None,
            num_clusters=num_covariate_clusters,
            save_raw_graph=None,
            save_clustered_graph=None,
            visualize_formed=None,
            shape_file=None,
            silent=True,
        )

        # Perform target variance clustering
        complete_target, target_clusters = perform_target_variance_clustering(
            target_data_frame=new_train_target_data_frame,
            num_clusters=num_target_clusters,
            silent=True,
        )

        # Now, get the right slice for the covariates and targets
        date_from, date_to, freq = new_train_subsampling_slice.split(":")
        subsample_indexer = pd.date_range(date_from, date_to, freq=f"{freq}d")

        # Since the covariates in the variance clustered graph are not subsampled,
        # we have to do it manually.
        # For this, we first get the date range over which the covariates are defined
        complete_timerange = (
            new_train_covariate_data_frames[0]
            .index.get_level_values("start_date")
            .unique()
            .sort_values()
        )
        for df in new_train_covariate_data_frames[1:]:
            complete_timerange = complete_timerange.intersection(
                df.index.get_level_values("start_date").unique().sort_values()
            )

        for node in graph.nodes:
            data = graph.nodes[node]["cov"]
            data = pd.DataFrame(data, index=complete_timerange)
            data = data.loc[subsample_indexer].to_numpy(copy=True)  # Subsampling step
            graph.nodes[node]["cov"] = data

        complete_target = complete_target.loc[subsample_indexer.shift(28, freq="d")]

        setattr(graph, "target_clusters", target_clusters)
        setattr(graph, "complete_target", complete_target.to_numpy(copy=True))

    skill, _, _, _ = location_wise_metric(
        new_valid_target_data_frame, new_valid_covariate_data_frames, std_data_frame, graph, "skill"
    )
    cos_sim, _, _, _ = location_wise_metric(
        new_valid_target_data_frame,
        new_valid_covariate_data_frames,
        std_data_frame,
        graph,
        "cosine-sim",
    )
    with open(log_file, "a") as f:
        f.write(f"{num_covariate_clusters} {num_target_clusters} {reg_param} {skill} {cos_sim}\n")


def run_multiple_configurations(
    new_train_covariate_data_frames,
    new_train_target_data_frame,
    new_valid_covariate_data_frames,
    new_valid_target_data_frame,
    new_train_subsampling_slice,
    std_data_frame,
    num_covariate_clusters_list,
    num_target_clusters_list,
    reg_param_list,
    clustering_algo,
    log_file,
):
    for num_covariate_clusters, num_target_clusters, reg_param in product(
        num_covariate_clusters_list, num_target_clusters_list, reg_param_list
    ):
        run_one_configuration(
            new_train_covariate_data_frames,
            new_train_target_data_frame,
            new_valid_covariate_data_frames,
            new_valid_target_data_frame,
            new_train_subsampling_slice,
            std_data_frame,
            num_covariate_clusters,
            num_target_clusters,
            reg_param,
            clustering_algo,
            log_file,
        )


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
        "--target_data_file_unstandardized",
        required=True,
        type=str,
        help="Specify the target data file (in .h5 format)",
    )
    p.add_argument(
        "--std_data_file", required=True, type=str, help="Specify the std data file (in .h5 format)"
    )
    p.add_argument(
        "--load_directory", required=True, type=str, help="Specify the loading directory"
    )
    p.add_argument(
        "--subsampling_slice",
        type=str,
        default="1981-09-26:2010-11-23:15",
        help="Specify the slice (in begin_date:end_date:freq format)",
    )
    p.add_argument(
        "--num_covariate_clusters_choices",
        type=int,
        nargs="+",
        required=True,
        help="Choices for number of covariate clusters",
    )
    p.add_argument(
        "--num_target_clusters_choices",
        type=int,
        nargs="+",
        required=True,
        help="Choices for number of target clusters",
    )
    p.add_argument(
        "--reg_param_choices",
        type=float,
        nargs="+",
        required=True,
        help="Choices for regularization parameter",
    )
    p.add_argument(
        "--log_file", type=str, default="logger.txt", help="File to log validation performance"
    )
    p.add_argument(
        "--clustering_algo",
        type=str,
        choices=["predictive", "variance"],
        required=True,
        help="Algorithm to perform validation grid search",
    )
    p.add_argument(
        "--begin_valid_year",
        type=int,
        default=1996,
        help="Provide beginning year for validation split",
    )
    args = p.parse_args()

    covariate_data_frames = []
    for data_file in args.covariate_data_files:
        covariate_data_frames.append(pd.read_hdf(os.path.join(args.load_directory, data_file)))

    target_data_frame = pd.read_hdf(os.path.join(args.load_directory, args.target_data_file))
    target_data_frame_unstandardized = pd.read_hdf(
        os.path.join(args.load_directory, args.target_data_file_unstandardized)
    )
    std_data_frame = pd.read_hdf(os.path.join(args.load_directory, args.std_data_file), key="std")

    required_data = construct_required_data(
        covariate_data_frames,
        target_data_frame,
        target_data_frame_unstandardized,
        args.subsampling_slice,
        args.begin_valid_year,
    )

    run_multiple_configurations(
        required_data[0],
        required_data[1],
        required_data[2],
        required_data[3],
        required_data[4],
        std_data_frame,
        args.num_covariate_clusters_choices,
        args.num_target_clusters_choices,
        args.reg_param_choices,
        args.clustering_algo,
        args.log_file,
    )
