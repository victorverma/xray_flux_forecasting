import argparse
import numpy as np
import os
import pandas as pd
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import get_context
from plotnine import *
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import scale
from typing import Optional

def estimate_precision_mat(df: pd.DataFrame, return_partial_cors=False) -> pd.DataFrame:
    """
    Estimate a precision matrix by using the graphical lasso on a given dataset.

    :param df: Pandas DataFrame representing the dataset.
    :param return_partial_cors: Boolean indicating whether to scale the precision matrix entries to be between -1 and 1.
    :return: A Pandas DataFrame representing the estimated precision matrix.
    """
    model = GraphicalLassoCV(assume_centered=True)
    model.fit(scale(df))
    precision_mat = model.precision_
    # See the following for information on partial correlations:
    # https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
    # https://stats.stackexchange.com/a/310935/58497
    if return_partial_cors:
        sqrt_diag = np.sqrt(np.diagonal(precision_mat))
        precision_mat = precision_mat / sqrt_diag[np.newaxis, :]
        precision_mat = precision_mat / sqrt_diag[:, np.newaxis]
        precision_mat = -precision_mat
    precision_mat = pd.DataFrame(precision_mat, index=df.columns, columns=df.columns)
    return precision_mat

def help_process_block(
        block_start_time: datetime,
        block_end_time: datetime,
        min_num_recs: int,
        return_partial_cors: bool,
        block: pd.DataFrame,
        mask: pd.Series
    ) -> Optional[pd.DataFrame]:
    """
    Estimate a precision matrix using a particular subset of a given block of data.

    :param block_start_time: Datetime giving the start of the block.
    :param block_end_time: Datetime giving the end of the block.
    :param min_num_recs: Integer giving the minimum number of records needed for precision matrix estimation.
    :param return_partial_cors: Boolean indicating whether to scale the precision matrix entries to be between -1 and 1.
    :param block: Pandas DataFrame representing the block.
    :param mask: Boolean Pandas Series indicating which rows of the block belong to the subset.
    :return: A Pandas DataFrame representing the estimated precision matrix.
    """
    num_recs = mask.sum()
    if num_recs >= min_num_recs:
        try:
            precision_mat = estimate_precision_mat(block[mask], return_partial_cors)
            precision_mat.attrs = {"block_start_time": block_start_time, "block_end_time": block_end_time, "num_recs": num_recs}
        except Exception as e:
            precision_mat = None
            print(f"Error for block_start_time = {block_start_time}, block_end_time = {block_end_time}: {e}.")
    else:
        precision_mat = None
        warnings.warn(
            f"Warning for block_start_time = {block_start_time}, block_end_time = {block_end_time}: record count is below min_num_recs.",
            UserWarning
        )
    return precision_mat

def process_block(
        block_start_time: datetime,
        block_end_time: datetime,
        min_num_recs: int,
        return_partial_cors: bool
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Estimate precision matrices using the non-flare and flare subsets of a given block of data.

    :param block_start_time: Datetime giving the start of the block.
    :param block_end_time: Datetime giving the end of the block.
    :param min_num_recs: Integer giving the minimum number of records needed for precision matrix estimation.
    :param return_partial_cors: Boolean indicating whether to scale the precision matrix entries to be between -1 and 1.
    :return: A tuple of two Pandas DataFrames that represent the estimated precision matrices for the non-flare and flare subsets.
    """
    block = harp_flare_data.loc[
        (harp_flare_data["T_REC"] >= block_start_time) & (harp_flare_data["T_REC"] <= block_end_time),
        predictor_cols + ["are_any_predictors_na", "was_during_flare"]
    ].copy()
    no_flare_mask = ~block["are_any_predictors_na"] & ~block["was_during_flare"]
    flare_mask = ~block["are_any_predictors_na"] & block["was_during_flare"]
    block.drop(columns=["are_any_predictors_na", "was_during_flare"], inplace=True)
    
    no_flare_precision_mat = help_process_block(block_start_time, block_end_time, min_num_recs, return_partial_cors, block, no_flare_mask)
    flare_precision_mat = help_process_block(block_start_time, block_end_time, min_num_recs, return_partial_cors, block, flare_mask)
    
    return no_flare_precision_mat, flare_precision_mat

def wrap_process_block(block_times: tuple[datetime, datetime]) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    block_start_time, block_end_time = block_times
    return process_block(block_start_time, block_end_time, min_num_recs, return_partial_cors)

def turn_attrs_into_cols(precision_mat: pd.DataFrame) -> pd.DataFrame:
    precision_mat = precision_mat.copy()
    for j, attr_and_val in enumerate(precision_mat.attrs.items()):
        attr, val = attr_and_val
        precision_mat.insert(j, attr, val)
    return precision_mat

def save_precision_mats(precision_mats: list[Optional[pd.DataFrame]], path: str) -> None:
    precision_mats = [turn_attrs_into_cols(precision_mat) for precision_mat in precision_mats if precision_mat is not None]
    if precision_mats:
        combined_precision_mats = pd.concat(precision_mats)
        # When len(precision_mats) == 1, combined_precision_mats == precision_mats[0], with attributes being preserved. However, Timestamp is not
        # JSON serializable, so we remove the attributes
        combined_precision_mats.attrs = {}
        combined_precision_mats.to_parquet(path)
    else:
        print(f"{os.path.basename(path)} couldn't be created as no precision matrices could be estimated.")

def plot_precision_mat(precision_mat: pd.DataFrame, fix_precision_lims: bool = False) -> ggplot:
    """
    Make a tile plot of a precision matrix; the fill color of each tile represents the value of the corresponding matrix entry.

    :param precision_mat: Pandas DataFrame that represents a precision matrix.
    :param fix_precision_lims: Boolean indicating whether to fix the fill color scale limits at -1 and 1.
    :return: ggplot that represents the tile plot.
    """
    if precision_mat is None:
        plot = None
    else:
        precision_mat_long = precision_mat.reset_index(names="col1").melt(id_vars="col1", var_name="col2", value_name="precision")
        limits = (-1, 1) if fix_precision_lims else None
        block_start_time = precision_mat.attrs["block_start_time"]
        block_end_time = precision_mat.attrs["block_end_time"]
        title = f"{block_start_time.tz_localize(None)}-{block_end_time.tz_localize(None)}"
        plot = (
            ggplot(precision_mat_long, aes(x="col1", y="col2", fill="precision")) +
            geom_tile() +
            scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0, limits=limits) +
            labs(x="Variable 1", y="Variable 2", fill="Precision", title=title) +
            theme_bw() +
            theme(axis_text_x=element_text(rotation=45))
        )
    return plot

def make_and_save_plots(precision_mats: list[Optional[pd.DataFrame]], filename: str, fix_precision_lims: bool) -> None:
    precision_mat_plots = [
        plot_precision_mat(precision_mat, fix_precision_lims=fix_precision_lims) for precision_mat in precision_mats if precision_mat is not None
    ]
    if precision_mat_plots:
        save_as_pdf_pages(precision_mat_plots, filename)
    else:
        print(f"{os.path.basename(filename)} couldn't be created as no precision matrices could be estimated.")

if __name__ == "__main__":
    print("Estimating precision matrices", end="", flush=True)
    script_start_time = time.time()

    ################################################################################
    # Parse the command-line arguments
    ################################################################################

    parser = argparse.ArgumentParser(description="Estimate predictor vector precision matrices using different blocks of data")
    parser.add_argument("--block_size", type=int, help="Block size (if set, also set --stride)")
    parser.add_argument("--stride", type=int, help="Stride length (if set, also set --block_size)")
    parser.add_argument("--predictors_str", type=str, required=True, help="Name of set of predictor columns to use")
    parser.add_argument("--flare_classes_str", type=str, required=True, help="Flare classes that define flaring status")
    parser.add_argument("--min_num_recs", type=int, required=True, help="Minimum number of records for use of the graphical lasso")
    parser.add_argument("--return_partial_cors", action="store_true", help="Whether to compute partial correlations (default: False)")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker processes to use")
    parser.add_argument("--chunksize", type=int, default=1, help="Chunk size for ProcessPoolExecutor instance")

    cmd_args = parser.parse_args()

    block_size = cmd_args.block_size
    stride = cmd_args.stride
    if (block_size is None) != (stride is None):
        parser.error("Either set both --block_size and --stride or set neither.")

    predictors_str = cmd_args.predictors_str
    flare_classes_str = cmd_args.flare_classes_str
    min_num_recs = cmd_args.min_num_recs
    return_partial_cors = cmd_args.return_partial_cors
    max_workers = cmd_args.max_workers
    chunksize = cmd_args.chunksize

    ################################################################################
    # Load and modify the data
    ################################################################################

    harp_flare_data = pd.read_parquet("../combined_data/harp_flare_data.parquet")

    sharp_params = [
        "USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZD", "TOTUSJZ", "MEANALP",
        "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT", "MEANSHR", "SHRGT45"
    ]
    areas_counts = ["NPIX", "SIZE", "AREA", "NACR", "SIZE_ACR", "AREA_ACR"] # Patch areas and pixel counts
    all_predictors = sharp_params + areas_counts

    predictor_cols_mapping = {"sharp_params": sharp_params, "areas_counts": areas_counts, "all_predictors": all_predictors}
    predictor_cols = predictor_cols_mapping[predictors_str]

    harp_flare_data["are_any_predictors_na"] = harp_flare_data[predictor_cols].isna().any(axis=1)

    flare_classes_mapping = {"a_plus": ["A", "B", "C", "M", "X"], "b_plus": ["B", "C", "M", "X"], "c_plus": ["C", "M", "X"], "m_plus": ["M", "X"]}
    flare_classes = flare_classes_mapping[flare_classes_str]
    harp_flare_data["was_during_flare"] = harp_flare_data["flare_class"].isin(flare_classes)

    harp_flare_data = harp_flare_data[["HARPNUM", "T_REC"] + predictor_cols + ["are_any_predictors_na", "was_during_flare"]]

    ################################################################################
    # Compute the block start and end times
    ################################################################################

    times = sorted(harp_flare_data["T_REC"].unique())
    num_times = len(times)
    if block_size is None:
        block_start_times = [times[0]]
        block_end_times = [times[-1]]
    elif block_size < num_times:
        block_start_times = [times[i] for i in range(0, num_times - block_size + 1, stride)]
        block_end_times = [times[i + block_size - 1] for i in range(0, num_times - block_size + 1, stride)]
    else:
        warnings.warn(f"block_size ({block_size}) is greater than or equal to the number of distinct times ({num_times}).", UserWarning)
        block_start_times = [times[0]]
        block_end_times = [times[-1]]

    ################################################################################
    # Estimate precision matrices
    ################################################################################

    output_tuples = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("fork")) as executor:
        output_tuples = list(executor.map(wrap_process_block, zip(block_start_times, block_end_times), chunksize=chunksize))
    no_flare_precision_mats, flare_precision_mats = zip(*output_tuples)

    ################################################################################
    # Save the results
    ################################################################################

    block_size = f"{block_size}_" if block_size is not None else ""
    stride = f"{stride}_" if stride is not None else ""
    return_partial_cors = "_partial_cors" if return_partial_cors else ""
    dir_name = f"{block_size}{stride}{predictors_str}_{flare_classes_str}_{min_num_recs}{return_partial_cors}"
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

    save_precision_mats(no_flare_precision_mats, os.path.join(dir_name, "no_flare_precision_mats.parquet"))
    make_and_save_plots(no_flare_precision_mats, os.path.join(dir_name, "no_flare_precision_plots.pdf"), fix_precision_lims=True)
    save_precision_mats(flare_precision_mats, os.path.join(dir_name, "flare_precision_mats.parquet"))
    make_and_save_plots(flare_precision_mats, os.path.join(dir_name, "flare_precision_plots.pdf"), fix_precision_lims=True)

    script_elapsed_time = time.time() - script_start_time
    print(f"\rEstimating precision matrices ({int(script_elapsed_time)}s)", flush=True)
    print("Done")
