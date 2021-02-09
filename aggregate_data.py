import os
import re
import sys
import calendar
import pandas as pd

from argparse import ArgumentParser


def mean_date_wise(data_files, load_directory, number_of_days):
    """
    Function to return the rolling average over `number_of_days` given a collection of
    data_files.

    Arguments:
    - data_files: list of file names of the collection of data (.h5 format).
    - load_directory: location where `data_files` are located.
    - number_of_days: number of days to take a rolling average over.
    """
    all_data = []
    for data_file in data_files:
        data = pd.read_hdf(os.path.join(load_directory, data_file))
        data = (
            data.reset_index(["lat", "lon"])
            .groupby(["lat", "lon"], group_keys=False)
            .resample("D")
            .ffill()
            .reset_index()
            .set_index(["lat", "lon", "start_date"])
        )  # This fixes the contiguity in dates
        all_data.append(data)
    all_data = pd.concat(all_data)
    all_data = (
        all_data.reset_index(["lat", "lon"])
        .groupby(["lat", "lon"], group_keys=False)
        .resample("D")
        .ffill()
        .reset_index()
        .set_index(["lat", "lon", "start_date"])
    )  # This fixes inter-year contiguity in dates
    all_data = all_data.unstack(level=["lat", "lon"])  # This way we can index date directly
    all_data.columns = all_data.columns.droplevel(0)  # This removes the redundant column index
    mean_rolling = all_data.rolling(number_of_days).mean()
    return mean_rolling.iloc[number_of_days - 1 :]  # Kill the NaNs in the top part


def get_mean_std_by_day(rolling_mean):
    """
    Function to return the mean of rolling averages across multiple years at
    all days of the year.

    Arguments:
    - rolling_mean: input from mean_date_wise
    """
    DUMMY_YEAR = 1904
    rolling_mean_copy = rolling_mean.copy()
    rolling_mean_copy.index = rolling_mean_copy.index.map(lambda idx: idx.replace(year=DUMMY_YEAR))
    mean = rolling_mean_copy.mean(level="start_date").sort_index().T.stack(level="start_date")
    std = rolling_mean_copy.std(level="start_date").sort_index().T.stack(level="start_date")
    return (mean, std)


def get_anomalies(
    data_files,
    load_directory,
    number_of_days,
    standardize=True,
    clustering_layout=True,
    mean_std=None,
):
    """
    Function to get the anomalies data based on a collection of data files for a given
    climate variable.

    Arguments:
    - data_files: list of file names of the collection of data (.h5 format).
    - load_directory: location where `data_files` are located.
    - number_of_days: number of days to take a rolling average over.
    - standardize: using mean and standard deviation as computed by `get_mean_std_by_day`.
                   standardize the daily anomalies. (default: True)
    - clustering_layout: if being passed to `construct_graph`, set to True.
                         (default: True)
    - mean_std: a tuple of pre-computed climatology (mean and std) (default: None)
    """
    rolling_mean = mean_date_wise(data_files, load_directory, number_of_days)
    if mean_std is None:
        mean_raw, std_raw = get_mean_std_by_day(rolling_mean)
    else:
        mean_raw, std_raw = mean_std
    mean = mean_raw.unstack(level=["lat", "lon"])
    std = std_raw.unstack(level=["lat", "lon"])
    for year in rolling_mean.index.year.unique():
        if calendar.isleap(year):
            mean_year = mean
        else:
            mean_year = mean[~((mean.index.day == 29) & (mean.index.month == 2))]
        mean_year.index = mean_year.index.map(lambda idx: idx.replace(year=year))
        rolling_mean[rolling_mean.index.year == year] -= mean_year

        if standardize:
            if calendar.isleap(year):
                std_year = std
            else:
                std_year = std[~((std.index.day == 29) & (std.index.month == 2))]
            std_year.index = std_year.index.map(lambda idx: idx.replace(year=year))
            rolling_mean[rolling_mean.index.year == year] /= std_year

    if clustering_layout:
        rolling_mean = (
            rolling_mean.stack(["lat", "lon"])
            .reorder_levels(["lat", "lon", "start_date"])
            .sort_index()
        )

    if mean_std is None:
        return (rolling_mean, (mean_raw, std_raw))
    else:
        return (rolling_mean, None)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--data_files",
        nargs="+",
        default=None,
        type=str,
        help="Specify the files to aggregate over",
    )
    p.add_argument(
        "--load_directory",
        required=True,
        type=str,
        help="Specify the location of the data_files to read from",
    )
    p.add_argument(
        "--variable", default=None, type=str, help="Specify the variables for aggregation"
    )
    p.add_argument(
        "--split",
        default=None,
        choices=["train", "test"],
        type=str,
        help="Specify the split for the variables (ignored if data_files is specified",
    )
    p.add_argument(
        "--save_directory",
        required=True,
        type=str,
        help="Specify the location to save the aggregated",
    )
    p.add_argument(
        "--save_name",
        required=True,
        type=str,
        help="Specify the name of the file to save the anomalies with",
    )
    p.add_argument(
        "--number_of_days",
        default=14,
        type=int,
        help="Number of days to consider a rolling average for",
    )
    p.add_argument(
        "--mean_std_file",
        default=None,
        type=str,
        help="Specify the mean and std file to read / write to",
    )
    p.add_argument("--no_standardize", action="store_true", help="Toggle to only center data")
    args = p.parse_args()

    if args.data_files is None and args.variable is None:
        raise ValueError("data_files and variable cannot be None simultaneously")

    if args.mean_std_file is not None:
        if os.path.exists(os.path.join(args.save_directory, args.mean_std_file)):
            mean = pd.read_hdf(os.path.join(args.save_directory, args.mean_std_file), key="mean")
            std = pd.read_hdf(os.path.join(args.save_directory, args.mean_std_file), key="std")
            mean_std = (mean, std)
        else:
            mean_std = None
    else:
        mean_std = None

    if args.data_files is not None:
        anomalies, mean_std = get_anomalies(
            args.data_files, args.load_directory, args.number_of_days, mean_std=mean_std
        )
        anomalies.to_hdf(os.path.join(args.save_directory, args.save_name), key="df")
        if mean_std is not None and args.mean_std_file is not None:
            mean_std[0].to_hdf(os.path.join(args.save_directory, args.mean_std_file), key="mean")
            mean_std[1].to_hdf(os.path.join(args.save_directory, args.mean_std_file), key="std")

    else:
        if not args.variable in ["hgt500", "rhum.sig995", "sst", "slp", "icec", "tmp2m", "sm"]:
            raise ValueError("Invalid variable specified")

        regex = re.compile(f"{args.variable}\.[0-9]{{4}}[\.a-z_]*\.h5")
        data_files = list(
            sorted([f for f in os.listdir(args.load_directory) if regex.match(f) is not None])
        )
        if args.split == "train":
            data_files = data_files[:-8]  # 1981 - 2010
        elif args.split == "test":
            data_files = data_files[-8:]  # 2011 - 2018

        anomalies, mean_std = get_anomalies(
            data_files,
            args.load_directory,
            args.number_of_days,
            mean_std=mean_std,
            standardize=not args.no_standardize,
        )
        anomalies.to_hdf(os.path.join(args.save_directory, args.save_name), key="df")
        if mean_std is not None and args.mean_std_file is not None:
            mean_std[0].to_hdf(os.path.join(args.save_directory, args.mean_std_file), key="mean")
            mean_std[1].to_hdf(os.path.join(args.save_directory, args.mean_std_file), key="std")
