import calendar
import numpy as np
import pandas as pd
import networkx as nx

from regression_utils import (
    skill_with_preds,
    multiple_regress,
    multiple_regress_l1,
    multiple_regress_l2,
    cosine_similarity_with_preds,
)


def location_wise_metric(
    target_data_frame, covariate_data_frame, std_data_frame, graph_or_other, metric, **kwargs
):
    if metric == "skill":
        metric_func = skill_with_preds
    elif metric == "cosine-sim":
        metric_func = cosine_similarity_with_preds
    else:
        raise ValueError(f"Invalid metric: {metric}")

    if isinstance(graph_or_other, nx.OrderedGraph):
        # We use the concatenated data matrix to build the covariates
        covariate_clusters = list(graph_or_other.nodes)
        data_matrix = np.hstack(
            [
                graph_or_other.nodes[node]["cov"] / len(node)
                if isinstance(node[0], tuple)
                else graph_or_other.nodes[node]["cov"]
                for node in covariate_clusters
            ]
        )  # size n x d_x
        target_matrix = graph_or_other.complete_target  # size n x d_y
    elif isinstance(graph_or_other, dict):
        if not (graph_or_other["type"] in ["multi_task_lasso", "xgboost"]):
            data_matrix = graph_or_other["covariates"]
            target_matrix = graph_or_other["complete_target"]

    # Get solution
    if not (graph_or_other["type"] in ["multi_task_lasso", "xgboost"]):
        if kwargs.get("reg_type") is not None:
            reg_type = kwargs.pop("reg_type")
            reg_param = kwargs.pop("reg_param") if kwargs.get("reg_param") is not None else 0.1
            if reg_type == "l1":
                w, b = multiple_regress_l1(data_matrix, target_matrix, reg_param)
            elif reg_type == "l2":
                w, b = multiple_regress_l2(data_matrix, target_matrix, reg_param)
            else:
                raise ValueError(f"Invalid reg_type {reg_type}")
        else:
            w, b = multiple_regress(data_matrix, target_matrix)

        regressor_function = lambda x: x @ w.T + b
    else:
        regressor_function = graph_or_other["model"]

    if kwargs.get("save_weights") is not None:
        np.save(kwargs.pop("save_weights"), w)

    if isinstance(graph_or_other, nx.OrderedGraph):
        data_matrix = []
        for cluster in covariate_clusters:
            if isinstance(cluster[0], tuple):
                cov = None
                for node in cluster:
                    if cov is None:
                        cov = covariate_data_frame.loc[node].to_numpy(copy=True)
                    else:
                        cov += covariate_data_frame.loc[node].to_numpy(copy=True)
                data_matrix.append(
                    cov / len(cluster)
                )  # size n_test x v, where v is the number of climate variables
            else:
                data_matrix.append(covariate_data_frame.loc[cluster].to_numpy(copy=True))
        data_matrix = np.hstack(data_matrix)  # size n_test x d_x
    else:
        if graph_or_other["type"] == "pca_variable_wise":
            data_matrix = []
            for idx, pca_obj in enumerate(graph_or_other["pca_objects"]):
                full_matrix = (
                    covariate_data_frame.iloc[:, idx].unstack(["lat", "lon"]).to_numpy(copy=True)
                )
                full_matrix = pca_obj.transform(full_matrix)
                data_matrix.append(full_matrix)
            data_matrix = np.hstack(data_matrix)  # size n_test x d_x

        elif graph_or_other["type"] == "pca_all_variables":
            data_matrix = [
                covariate_data_frame.iloc[:, idx].unstack(["lat", "lon"]).to_numpy(copy=True)
                for idx in range(covariate_data_frame.shape[1])
            ]
            data_matrix = np.hstack(data_matrix)
            data_matrix = graph_or_other["pca_objects"].transform(data_matrix)

        elif graph_or_other["type"] == "nino":
            data_matrix = covariate_data_frame.to_numpy(copy=True)

        elif graph_or_other["type"] in ["multi_task_lasso", "xgboost"]:
            data_matrix = [
                covariate_data_frame.iloc[:, idx].unstack(["lat", "lon"]).to_numpy(copy=True)
                for idx in range(covariate_data_frame.shape[1])
            ]
            data_matrix = np.hstack(data_matrix)

    raw_preds = regressor_function(data_matrix)  # size n_test x d_y
    # raw_preds is a DataFrame with index given the dates at which we have predicted
    raw_preds = pd.DataFrame(
        raw_preds,
        index=covariate_data_frame.index.get_level_values("start_date").unique().sort_values(),
    )

    # Now we construct the dataframe of standardized predictions at the location level
    if isinstance(graph_or_other, nx.OrderedGraph):
        target_clusters = graph_or_other.target_clusters
    else:
        target_clusters = graph_or_other["target_clusters"]
    preds_df = []
    for (node, _) in target_data_frame.groupby(level=[0, 1]):
        for (i, cluster) in enumerate(target_clusters):
            if isinstance(cluster[0], tuple):
                if node in cluster:
                    preds_df.append(
                        pd.DataFrame({"lat": node[0], "lon": node[1], "pred": raw_preds.iloc[:, i]})
                    )
                    break
            else:
                if node == cluster:
                    preds_df.append(
                        pd.DataFrame({"lat": node[0], "lon": node[1], "pred": raw_preds.iloc[:, i]})
                    )
                    break
    preds_df = (
        pd.concat(preds_df).reset_index().set_index(["lat", "lon", "start_date"]).sort_index()
    )
    preds_df = preds_df.unstack(level=["lat", "lon"])
    preds_df.columns = preds_df.columns.droplevel(0)  # Drop the redundant column index
    del raw_preds  # Save memory

    std = std_data_frame.unstack(level=["lat", "lon"])
    for year in preds_df.index.year.unique():
        if calendar.isleap(year):
            std_year = std
        else:
            std_year = std[~((std.index.day == 29) & (std.index.month == 2))]
        std_year.index = std_year.index.map(lambda idx: idx.replace(year=year))
        preds_df[preds_df.index.year == year] *= std_year

    preds_df.index = preds_df.index.shift(28, freq="d")  # Shifting to match target index
    preds_df = (
        preds_df.stack(["lat", "lon"]).reorder_levels(["lat", "lon", "start_date"]).sort_index()
    )
    assert preds_df.index.equals(target_data_frame.index), "Indices don't match"

    if kwargs.get("save_preds") is not None:
        preds_df.to_hdf(kwargs.pop("save_preds"), key="df")

    location_wise_metric_df = pd.DataFrame(columns=["lat", "lon", "metric"])
    for (node, target), (node_check, predict) in zip(
        target_data_frame.groupby(level=[0, 1]), preds_df.groupby(level=[0, 1])
    ):
        assert node == node_check, "Node mismatch"
        metric_val = metric_func(
            target.to_numpy().reshape(-1, 1), predict.to_numpy().reshape(-1, 1)
        )
        location_wise_metric_df = location_wise_metric_df.append(
            {"lat": node[0], "lon": node[1], "metric": metric_val}, ignore_index=True
        )

    # Now we do the cluster levels evaluation
    # At each cluster, we take the mean of prediction and target series to represent the prediction
    # for that cluster. These are then directly compared.
    coloring = {}
    cluster_pred_map = {cluster: None for cluster in target_clusters}
    cluster_target_map = {cluster: None for cluster in target_clusters}

    for (node, target), (node_check, predict) in zip(
        target_data_frame.groupby(level=[0, 1]), preds_df.groupby(level=[0, 1])
    ):
        predict_val = predict.to_numpy(copy=True)
        target_val = target.to_numpy(copy=True)
        assert node == node_check, "Node mismatch"
        for cluster in target_clusters:
            if isinstance(cluster[0], tuple):
                if node in cluster:
                    if cluster_pred_map[cluster] is None:
                        cluster_pred_map[cluster] = predict_val
                        cluster_target_map[cluster] = target_val
                    else:
                        cluster_pred_map[cluster] += predict_val
                        cluster_target_map[cluster] += target_val
                    break
            else:
                if node == cluster:
                    if cluster_pred_map[cluster] is None:
                        cluster_pred_map[cluster] = predict_val
                        cluster_target_map[cluster] = target_val
                    else:
                        cluster_pred_map[cluster] += predict_val
                        cluster_target_map[cluster] += target_val
                    break

    for cluster in target_clusters:
        len_cluster = len(cluster) if isinstance(cluster[0], tuple) else 1
        coloring[cluster] = metric_func(
            cluster_target_map[cluster].squeeze() / len_cluster,
            cluster_pred_map[cluster].squeeze() / len_cluster,
        )

    return (
        metric_func(target_data_frame.to_numpy(copy=True), preds_df.to_numpy(copy=True)),
        location_wise_metric_df,
        coloring,
        target_clusters,
    )


def year_wise_metric(target_data_frame, preds_data_frame, metric):
    if metric == "skill":
        metric_func = skill_with_preds
    elif metric == "cosine-sim":
        metric_func = cosine_similarity_with_preds
    else:
        raise ValueError(f"Invalid metric: {metric}")

    assert target_data_frame.index.equals(preds_data_frame.index), "Indices don't match"
    target_data_frame_unstacked = target_data_frame.unstack(["lat", "lon"])
    preds_data_frame_unstacked = preds_data_frame.unstack(["lat", "lon"])
    unique_years = target_data_frame_unstacked.index.year.unique().tolist()
    complete_index = target_data_frame_unstacked.index

    metric_dict = dict()
    for year in unique_years:
        indexer = (complete_index <= f"{year}-12-31") & (complete_index >= f"{year}-01-01")
        indexer = complete_index[indexer]
        target_df_year = target_data_frame_unstacked.loc[indexer]
        preds_df_year = preds_data_frame_unstacked.loc[indexer]
        metric_dict[year] = metric_func(
            target_df_year.to_numpy(copy=True), preds_df_year.to_numpy(copy=True)
        )

    return metric_dict
