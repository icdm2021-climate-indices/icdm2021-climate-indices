import os
import numpy as np
import pandas as pd
import pickle as pkl

from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.decomposition import PCA

from multivariate_predictive_clustering import (
    perform_target_variance_clustering,
    create_subsampled_dataset,
)


def compute_pca(matrix, n_components):
    pca_obj = PCA(n_components=n_components)
    pca_obj.fit(matrix)
    return pca_obj


def pca_variable_wise_features_for_regression(covariate_data_frames, n_components, silent=False):
    pca_objects = []
    all_covariates = []
    for covariate_data_frame in tqdm(
        covariate_data_frames, ascii=True, desc="pca_compute", unit="covariate", disable=silent
    ):
        mat_form = covariate_data_frame.unstack(["lat", "lon"]).to_numpy(copy=True)
        pca = compute_pca(mat_form, n_components)
        mat_form = pca.transform(mat_form)
        all_covariates.append(mat_form)
        pca_objects.append(pca)
    all_covariates = np.hstack(all_covariates)
    pca_baseline_dict = dict()
    pca_baseline_dict["type"] = "pca_variable_wise"
    pca_baseline_dict["pca_objects"] = pca_objects
    pca_baseline_dict["covariates"] = all_covariates
    return pca_baseline_dict


def pca_all_variables_features_for_regression(covariate_data_frames, n_components):
    all_covariates = [
        cov_df.unstack(["lat", "lon"]).to_numpy(copy=True) for cov_df in covariate_data_frames
    ]
    all_covariates = np.hstack(all_covariates)
    pca = compute_pca(all_covariates, n_components)
    all_covariates = pca.transform(all_covariates)
    pca_baseline_dict = dict()
    pca_baseline_dict["type"] = "pca_all_variables"
    pca_baseline_dict["pca_objects"] = pca
    pca_baseline_dict["covariates"] = all_covariates
    return pca_baseline_dict


def preprocess_and_subsample(
    covariate_data_files,
    target_date_file,
    load_directory,
    subsampling_slice,
    num_target_clusters,
    preprocess_date_range,
    silent,
):
    covariate_data_frames = []
    for data_file in covariate_data_files:
        df = pd.read_hdf(os.path.join(load_directory, data_file))
        if preprocess_date_range is not None:
            date_from, date_to = preprocess_date_range.split(":")
            df = df.unstack(["lat", "lon"])
            df = df.loc[pd.date_range(date_from, date_to)]
            df.index.name = "start_date"
            df = df.stack(["lat", "lon"]).reorder_levels(["lat", "lon", "start_date"]).sort_index()
        covariate_data_frames.append(df)

    target_data_frame = pd.read_hdf(os.path.join(load_directory, target_data_file))
    if preprocess_date_range is not None:
        date_from, date_to = preprocess_date_range.split(":")
        target_data_frame = target_data_frame.unstack(["lat", "lon"])
        target_data_frame = target_data_frame.loc[pd.date_range(date_from, date_to)]
        target_data_frame.index.name = "start_date"
        target_data_frame = (
            target_data_frame.stack(["lat", "lon"])
            .reorder_levels(["lat", "lon", "start_date"])
            .sort_index()
        )

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
    return covariate_data_frames, target_data_frame, target_clusters


def run_pca_baseline(
    covariate_data_files,
    target_data_file,
    load_directory,
    save_name,
    subsampling_slice,
    num_covariate_components,
    num_target_clusters,
    variable_wise,
    preprocess_date_range,
    silent,
):
    covariate_data_frames, target_data_frame, target_clusters = preprocess_and_subsample(
        covariate_data_files=covariate_data_files,
        target_data_file=target_data_file,
        load_directory=load_directory,
        subsampling_slice=subsampling_slice,
        num_target_clusters=num_target_clusters,
        preprocess_date_range=preprocess_date_range,
        silent=silent,
    )

    if variable_wise:
        pca_baseline_dict = pca_variable_wise_features_for_regression(
            covariate_data_frames=covariate_data_frames,
            n_components=num_covariate_components,
            silent=silent,
        )
    else:
        pca_baseline_dict = pca_all_variables_features_for_regression(
            covariate_data_frames=covariate_data_frames, n_components=num_covariate_components
        )

    pca_baseline_dict["target_clusters"] = target_clusters
    pca_baseline_dict["complete_target"] = target_data_frame.to_numpy(copy=True)
    pkl.dump(pca_baseline_dict, open(os.path.join(load_directory, save_name + ".pkl"), "wb"))


def run_nino_baseline(
    nino_data_file,
    target_data_file,
    load_directory,
    save_name,
    subsampling_slice,
    num_target_clusters,
    silent,
):
    nino_data_frame = pd.read_hdf(os.path.join(load_directory, nino_data_file), key="df")

    target_data_frame = pd.read_hdf(os.path.join(load_directory, target_data_file))
    target_data_frame, target_clusters = perform_target_variance_clustering(
        target_data_frame=target_data_frame, num_clusters=num_target_clusters, silent=silent
    )

    date_from, date_to, freq = subsampling_slice.split(":")
    subsample_indexer = pd.date_range(date_from, date_to, freq=f"{freq}d")
    nino_data_frame = nino_data_frame.loc[subsample_indexer]
    _, target_data_frame = create_subsampled_dataset(
        covariate_data_frames=[],
        target_data_frame=target_data_frame,
        date_from=date_from,
        date_to=date_to,
        freq=freq,
    )

    nino_baseline_dict = dict()
    nino_baseline_dict["type"] = "nino"
    nino_baseline_dict["covariates"] = nino_data_frame.to_numpy(copy=True)
    nino_baseline_dict["target_clusters"] = target_clusters
    nino_baseline_dict["complete_target"] = target_data_frame.to_numpy(copy=True)
    pkl.dump(nino_baseline_dict, open(os.path.join(load_directory, save_name + ".pkl"), "wb"))


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
        "--save_name",
        required=True,
        type=str,
        help="Specify the name of the file to save the regression data to",
    )
    p.add_argument(
        "--num_covariate_components",
        type=int,
        default=22,
        help="Specify the number of components per variable (covariate reduction)",
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
    p.add_argument("--silent", action="store_true", help="Toggle to mute progress bar")
    p.add_argument(
        "--baseline_type",
        type=str,
        required=True,
        choices=["pca_variable_wise", "pca_all_variables", "nino"],
        help="Specify the type of baseline",
    )
    p.add_argument(
        "--preprocess_date_range",
        default=None,
        type=str,
        help="Specify date range to consider (in begin_date:end_date format)",
    )
    args = p.parse_args()

    if args.baseline_type.find("pca") != -1:
        run_pca_baseline(
            covariate_data_files=args.covariate_data_files,
            target_data_file=args.target_data_file,
            load_directory=args.load_directory,
            save_name=args.save_name,
            subsampling_slice=args.subsampling_slice,
            num_covariate_components=args.num_covariate_components,
            num_target_clusters=args.num_target_clusters,
            variable_wise=(args.baseline_type == "pca_variable_wise"),
            preprocess_date_range=args.preprocess_date_range,
            silent=args.silent,
        )
    else:
        run_nino_baseline(
            nino_data_file=args.covariate_data_files[0],
            target_data_file=args.target_data_file,
            load_directory=args.load_directory,
            save_name=args.save_name,
            subsampling_slice=args.subsampling_slice,
            num_target_clusters=args.num_target_clusters,
            silent=args.silent,
        )
