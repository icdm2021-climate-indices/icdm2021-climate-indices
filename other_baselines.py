import os
import json
import numpy as np
import pandas as pd

from itertools import product
from argparse import ArgumentParser

from xgboost import XGBRegressor
from sklearn.linear_model import MultiTaskLasso
from sklearn.multioutput import MultiOutputRegressor

from baselines import compute_pca
from torch_utils import run_autoencoder
from evaluation_utils import location_wise_metric
from validation_grid_search import construct_required_data
from multivariate_predictive_clustering import perform_target_variance_clustering


def get_featurizer(covariate_matrix, method_and_parameters):
    """
    Function to get features of reduced dimensionality from a matrix of covariates.
    The columns indicate the values of a climate variable at a particular location,
    and therefore there are d_y x |M| of them.

    Arguments:
    - covariate_matrix: T x (d_y x |M|) np.ndarray
    - method_and_parameters: dictionary of configuration including name and parameters
    """
    if method_and_parameters["name"] == "autoencoder":
        featurizer = run_autoencoder(data=covariate_matrix, config=method_and_parameters["params"])

    elif method_and_parameters["name"] == "pca":
        pca_obj = compute_pca(
            covariate_matrix, n_components=method_and_parameters["params"]["n_components"]
        )
        featurizer = pca_obj.transform

    return featurizer


def run_one_configuration(
    full_train_covariate_matrix,
    complete_target,
    new_valid_covariate_data_frames,
    new_valid_target_data_frame,
    std_data_frame,
    target_clusters,
    featurizer,
    model_name,
    parameters,
    log_file,
):
    model_baseline = dict()
    model_baseline["type"] = model_name
    model_baseline["target_clusters"] = target_clusters

    if model_name == "multi_task_lasso":
        model = MultiTaskLasso(max_iter=5000, **parameters)
    elif model_name == "xgboost":
        model = MultiOutputRegressor(
            XGBRegressor(n_jobs=10, objective="reg:squarederror", verbosity=0, **parameters)
        )

    model.fit(featurizer(full_train_covariate_matrix), complete_target.to_numpy(copy=True))
    model_baseline["model"] = lambda x: model.predict(featurizer(x))

    skill, _, _, _ = location_wise_metric(
        new_valid_target_data_frame,
        new_valid_covariate_data_frames,
        std_data_frame,
        model_baseline,
        "skill",
    )
    cos_sim, _, _, _ = location_wise_metric(
        new_valid_target_data_frame,
        new_valid_covariate_data_frames,
        std_data_frame,
        model_baseline,
        "cosine-sim",
    )
    with open(log_file, "a") as f:
        f.write(f"{len(target_clusters)} {parameters} {skill} {cos_sim}\n")


def run_multiple_configurations(
    full_train_covariate_matrix,
    complete_target,
    new_valid_covariate_data_frames,
    new_valid_target_data_frame,
    std_data_frame,
    target_clusters,
    featurizer,
    model_name,
    parameters_list,
    log_file,
):
    param_names = list(parameters_list.keys())
    param_values = list(parameters_list.values())
    for param_settings in product(*param_values):
        parameters = {
            param_name: param_setting
            for (param_name, param_setting) in zip(param_names, param_settings)
        }
        run_one_configuration(
            full_train_covariate_matrix,
            complete_target,
            new_valid_covariate_data_frames,
            new_valid_target_data_frame,
            std_data_frame,
            target_clusters,
            featurizer,
            model_name,
            parameters,
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
        "--num_target_clusters", type=int, required=True, help="Number of target clusters"
    )
    p.add_argument(
        "--model_config_file",
        type=str,
        required=True,
        help="Config file to explicitly state the model and parameter grid",
    )
    p.add_argument(
        "--log_file", type=str, default="logger.txt", help="File to log validation performance"
    )
    p.add_argument(
        "--begin_valid_year",
        type=int,
        default=1996,
        help="Provide beginning year for validation split",
    )
    p.add_argument(
        "--featurize_transform",
        type=str,
        default=None,
        help="Config file to explicitly create features (PCA based / autoencoder based)",
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

    if args.featurize_transform is not None:
        featurize_transform = json.load(open(args.featurize_transform))
    else:
        featurize_transform = None

    complete_target, target_clusters = perform_target_variance_clustering(
        target_data_frame=required_data[1], num_clusters=args.num_target_clusters, silent=True
    )

    # Now, get the right slice for the covariates and targets
    date_from, date_to, freq = required_data[4].split(":")
    subsample_indexer = pd.date_range(date_from, date_to, freq=f"{freq}d")

    full_train_covariate_matrix = []
    for cov_df in required_data[0]:
        tmp = cov_df.unstack(["lat", "lon"]).loc[subsample_indexer]
        full_train_covariate_matrix.append(tmp.to_numpy(copy=True))
    full_train_covariate_matrix = np.hstack(full_train_covariate_matrix)  # T x (d_x * |M|)

    if featurize_transform is not None:
        featurizer = get_featurizer(full_train_covariate_matrix, featurize_transform)
    else:
        featurizer = lambda x: x

    complete_target = complete_target.loc[subsample_indexer.shift(28, freq="d")]

    model_spec = json.load(open(args.model_config_file))

    run_multiple_configurations(
        full_train_covariate_matrix,
        complete_target,
        required_data[2],
        required_data[3],
        std_data_frame,
        target_clusters,
        featurizer,
        model_spec["name"],
        model_spec["params"],
        args.log_file,
    )
