import argparse
import numpy as np
import os
import pandas as pd
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import get_context
from plotnine import *
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import scale
from typing import Optional

print("Estimating precision matrices", end="", flush=True)
script_start_time = time.time()

################################################################################
# Parse the command-line arguments
################################################################################

parser = argparse.ArgumentParser(description="Estimate predictor vector precision matrices using different blocks of data")
parser.add_argument("--block_size", type=int, required=True, help="Block size")
parser.add_argument("--stride", type=int, required=True, help="Stride length")
parser.add_argument("--predictor_cols_str", type=str, required=True, help="Name of set of predictor columns to use")
parser.add_argument("--flare_classes_str", type=str, required=True, help="Flare classes that define flaring status")
parser.add_argument("--min_num_recs", type=int, required=True, help="Minimum number of records for use of the graphical lasso")
parser.add_argument("--return_partial_cors", action="store_true", help="Whether to compute partial correlations (default: False)")
parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker processes to use")
parser.add_argument("--chunksize", type=int, default=1, help="Chunk size for ProcessPoolExecutor instance")

cmd_args = parser.parse_args()
block_size = cmd_args.block_size
stride = cmd_args.stride
predictor_cols_str = cmd_args.predictor_cols_str
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

harp_flare_data["are_any_sharp_params_na"] = harp_flare_data[sharp_params].isna().any(axis=1)
harp_flare_data["are_any_areas_counts_na"] = harp_flare_data[areas_counts].isna().any(axis=1)
harp_flare_data["are_any_predictors_na"] = harp_flare_data[all_predictors].isna().any(axis=1)
harp_flare_data["was_during_flare"] = ~harp_flare_data["flare_class"].isna()
harp_flare_data["was_during_c_plus_flare"] = harp_flare_data["flare_class"].isin(["C", "M", "X"])
harp_flare_data["was_during_m_plus_flare"] = harp_flare_data["flare_class"].isin(["M", "X"])

################################################################################
# Compute the block start and end times
################################################################################

times = sorted(harp_flare_data["T_REC"].unique())
if len(times) < block_size:
    block_start_times = []
    block_end_times = []
else:
    block_start_times = [times[i] for i in range(0, len(times), stride)]
    block_end_times = [times[min(i + block_size - 1, len(times) - 1)] for i in range(0, len(times), stride)]

################################################################################
# Define functions
################################################################################

def run_graphical_lasso(
        harp_data: pd.DataFrame,
        cols: list[str],
        alphas=4,
        n_refinements=4,
        cv=None,
        tol=0.0001,
        enet_tol=0.0001,
        max_iter=100,
        mode="cd",
        n_jobs=None,
        verbose=False,
        eps=np.float64(2.220446049250313e-16),
        return_partial_cors=False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    scale(harp_data, copy=False)
    model = GraphicalLassoCV(
        alphas=alphas,
        n_refinements=n_refinements,
        cv=cv,
        tol=tol,
        enet_tol=enet_tol,
        max_iter=max_iter,
        mode=mode,
        n_jobs=n_jobs,
        verbose=verbose,
        eps=eps,
        assume_centered=True
    )
    model.fit(harp_data)

    precision_mat = model.precision_
    # See the following for information on partial correlations:
    # https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
    # https://stats.stackexchange.com/a/310935/58497
    if return_partial_cors:
        sqrt_diag = np.sqrt(np.diagonal(precision_mat))
        precision_mat = precision_mat / sqrt_diag[np.newaxis, :]
        precision_mat = precision_mat / sqrt_diag[:, np.newaxis]
        precision_mat = -precision_mat
    precision_mat = pd.DataFrame(precision_mat).rename(columns={i: col for i, col in enumerate(cols)})
    precision_mat.index = cols

    costs = pd.DataFrame(model.costs_).rename(columns={0: "obj_fun_val", 1: "dual_gap"}).reset_index(names="iter_num")
    alpha = model.alpha_
    cv_results = model.cv_results_

    return precision_mat, costs, alpha, cv_results

def plot_precision_mat(precision_mat: pd.DataFrame, fix_precision_lims=False) -> ggplot:
    precision_mat_long = precision_mat.reset_index(names="var1").melt(id_vars="var1", var_name="var2", value_name="precision")
    limits = (-1, 1) if fix_precision_lims else None
    plot = (
        ggplot(precision_mat_long, aes(x="var1", y="var2", fill="precision")) +
        geom_tile() +
        scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0, limits=limits) +
        labs(x="Variable 1", y="Variable 2", fill="Precision") +
        theme_bw() +
        theme(axis_text_x=element_text(rotation=45))
    )
    return plot

def process_block(
        block_start_time: datetime,
        block_end_time: datetime,
        predictor_cols_str: str,
        flare_classes_str: str,
        min_num_recs: int,
        return_partial_cors: bool = False
    ) -> tuple[Optional[pd.DataFrame], Optional[ggplot], Optional[pd.DataFrame], Optional[ggplot]]:
    predictor_cols_str_mapping = {"sharp_params": sharp_params, "areas_counts": areas_counts, "all_predictors": all_predictors}
    predictor_cols = predictor_cols_str_mapping[predictor_cols_str]
    na_col_str_mapping = {
        "sharp_params": "are_any_sharp_params_na", "areas_counts": "are_any_areas_counts_na", "all_predictors": "are_any_predictors_na"
    }
    na_col = na_col_str_mapping[predictor_cols_str]
    flare_classes_str_mapping = {"a_plus": "was_during_flare", "c_plus": "was_during_c_plus_flare", "m_plus": "was_during_m_plus_flare"}
    flare_classes_col = flare_classes_str_mapping[flare_classes_str]
    block = harp_flare_data.loc[
        (harp_flare_data["T_REC"] >= block_start_time) & (harp_flare_data["T_REC"] <= block_end_time),
        predictor_cols + [na_col, flare_classes_col]
    ].copy()
    no_flare_recs_mask = ~block[na_col] & ~block[flare_classes_col]
    flare_recs_mask = ~block[na_col] & block[flare_classes_col]
    block.drop(columns=[na_col, flare_classes_col], inplace=True)

    plot_title = f"{block_start_time.tz_localize(None)}-{block_end_time.tz_localize(None)}"
    
    num_no_flare_recs = no_flare_recs_mask.sum()
    if num_no_flare_recs >= min_num_recs:
        try:
            no_flare_precision_mat, _, _, _ = run_graphical_lasso(
                block[no_flare_recs_mask], predictor_cols, return_partial_cors=return_partial_cors
            )
            no_flare_precision_mat_plot = plot_precision_mat(no_flare_precision_mat, fix_precision_lims=return_partial_cors)
            no_flare_precision_mat_plot = no_flare_precision_mat_plot + ggtitle(plot_title)
            no_flare_precision_mat["block_start_time"] = block_start_time
            no_flare_precision_mat["block_end_time"] = block_end_time
            no_flare_precision_mat["num_recs"] = num_no_flare_recs
        except Exception as e:
            no_flare_precision_mat = None
            no_flare_precision_mat_plot = None
            print(f"Error occurred: {e}. block_start_time = {block_start_time}, block_end_time = {block_end_time}.")
    else:
        no_flare_precision_mat = None
        no_flare_precision_mat_plot = None

    num_flare_recs = flare_recs_mask.sum()
    if num_flare_recs >= min_num_recs:
        try:
            flare_precision_mat, _, _, _ = run_graphical_lasso(
                block[flare_recs_mask], predictor_cols, return_partial_cors=return_partial_cors
            )
            flare_precision_mat_plot = plot_precision_mat(flare_precision_mat, fix_precision_lims=return_partial_cors)
            flare_precision_mat_plot = flare_precision_mat_plot + ggtitle(plot_title)
            flare_precision_mat["block_start_time"] = block_start_time
            flare_precision_mat["block_end_time"] = block_end_time
            flare_precision_mat["num_recs"] = num_flare_recs
        except Exception as e:
            flare_precision_mat = None
            flare_precision_mat_plot = None
            print(f"Error occurred: {e}. block_start_time = {block_start_time}, block_end_time = {block_end_time}.")
    else:
        flare_precision_mat = None
        flare_precision_mat_plot = None
    
    return no_flare_precision_mat, no_flare_precision_mat_plot, flare_precision_mat, flare_precision_mat_plot

def wrapper(block_times: tuple[datetime, datetime]) -> tuple[Optional[pd.DataFrame], Optional[ggplot], Optional[pd.DataFrame], Optional[ggplot]]:
    block_start_time, block_end_time = block_times
    return process_block(
        block_start_time,
        block_end_time,
        predictor_cols_str,
        flare_classes_str,
        min_num_recs,
        return_partial_cors
    )

################################################################################
# Estimate precision matrices
################################################################################

output_tuples = []
with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("fork")) as executor:
    output_tuples = list(
        executor.map(wrapper, zip(block_start_times, block_end_times), chunksize=chunksize)
    )
no_flare_precision_mats, no_flare_precision_mat_plots, flare_precision_mats, flare_precision_mat_plots = zip(*output_tuples)

################################################################################
# Save the results
################################################################################

return_partial_cors = "_partial_cors" if return_partial_cors else ""
dir_name = f"{block_size}_{stride}_{predictor_cols_str}_{flare_classes_str}_{min_num_recs}{return_partial_cors}"
if os.path.exists(dir_name):
    shutil.rmtree(dir_name)
os.mkdir(dir_name)

no_flare_precision_mats = [mat for mat in no_flare_precision_mats if mat is not None]
if no_flare_precision_mats:
    no_flare_precision_mats = pd.concat(no_flare_precision_mats)
    no_flare_precision_mats.to_parquet(os.path.join(dir_name, "no_flare_precision_mats.parquet"))
else:
    print("No no-flare precision matrices could be estimated.")

no_flare_precision_mat_plots = [plot for plot in no_flare_precision_mat_plots if plot is not None]
if no_flare_precision_mat_plots:
    save_as_pdf_pages(no_flare_precision_mat_plots, filename=os.path.join(dir_name, "no_flare_precision_mat_plots.pdf"))
else:
    print("No no-flare precision matrix plots could be generated.")

flare_precision_mats = [mat for mat in flare_precision_mats if mat is not None]
if flare_precision_mats:
    flare_precision_mats = pd.concat(flare_precision_mats)
    flare_precision_mats.to_parquet(os.path.join(dir_name, "flare_precision_mats.parquet"))
else:
    print("No flare precision matrices could be estimated.")

flare_precision_mat_plots = [plot for plot in flare_precision_mat_plots if plot is not None]
if flare_precision_mat_plots:
    save_as_pdf_pages(flare_precision_mat_plots, filename=os.path.join(dir_name, "flare_precision_mat_plots.pdf"))
else:
    print("No flare precision matrix plots could be generated.")

script_elapsed_time = time.time() - script_start_time
print(f"\rEstimating precision matrices ({int(script_elapsed_time)}s)", flush=True)
print("Done")
