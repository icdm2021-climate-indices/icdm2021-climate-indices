import os
import re
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser


def mask_data(data_files, mask_file, load_directory, save_directory):
    """
    Function to impose a location mask on a collection of data and save a copy.

    Arguments:
    - data_files: list of file names of the collection of data (.h5 format).
    - mask_file: file name of the mask (.h5 format).
    - load_directory: location where `mask_file` and `data_file` are located.
    - save_directory: location to save the masked data.
    """
    mask_df = pd.read_hdf(os.path.join(args.load_directory, args.mask_file))
    for data_file in tqdm(data_files, leave=False, ascii=True):
        complete_df = pd.read_hdf(os.path.join(load_directory, data_file))
        complete_df = complete_df.reset_index()  # Reset index just to be sure
        complete_df = pd.merge(complete_df, mask_df, how="inner", on=["lat", "lon"])
        complete_df.set_index(["lat", "lon", "start_date"], inplace=True)
        complete_df.to_hdf(
            os.path.join(save_directory, data_file[:-3] + f".{os.path.basename(mask_file)}"),
            key="df",
        )


def generate_masked_data(variables, mask_file, load_directory, save_directory):
    """
    Function to create masked datasets for collection of climate variables.

    Arguments:
    - variables: list of climate variables, with original naming convention.
    - mask_file: file name of the mask (.h5 format).
    - load_directory: location where `mask_file` and `data_file` are located.
    - save_directory: location to save the masked data.
    """
    if not set(variables).issubset(["hgt500", "rhum.sig995", "sst", "slp", "icec", "tmp2m", "sm"]):
        raise ValueError("Invalid variable set specified")

    for variable in tqdm(set(variables), ascii=True):
        regex = re.compile(f"{variable}\.[0-9]{{4}}\.h5")
        data_files = [f for f in os.listdir(load_directory) if regex.match(f) is not None]
        mask_data(data_files, mask_file, load_directory, save_directory)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--mask_file", required=True, help="Specify mask file (.h5 format)")
    p.add_argument(
        "--data_files",
        nargs="+",
        default=None,
        type=str,
        help="Specify the data files over which you would like to impose the mask on (.h5 format)",
    )
    p.add_argument(
        "--load_directory",
        required=True,
        type=str,
        help="Specify the location of the data_files to read from",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        default=None,
        type=str,
        help="Specify the variables that you would like to impose the mask on (.h5 format)",
    )
    p.add_argument(
        "--save_directory",
        required=True,
        type=str,
        help="Specify the location of the masked data_files to save to",
    )
    args = p.parse_args()

    if args.data_files is None and args.variables is None:
        raise ValueError("data_files and variables cannot be None simultaneously")

    if args.data_files is not None:
        mask_data(
            mask_file=args.mask_file,
            data_files=args.data_files,
            load_directory=args.load_directory,
            save_directory=args.save_directory,
        )
    else:
        generate_masked_data(
            variables=args.variables,
            mask_file=args.mask_file,
            load_directory=args.load_directory,
            save_directory=args.save_directory,
        )
