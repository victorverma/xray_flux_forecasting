import argparse
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

print("Matching flares to HARPs", end="", flush=True)
script_start_time = time.time()

################################################################################
# Parse the command-line arguments
################################################################################

parser = argparse.ArgumentParser(description="For each HARP record, find any matching flares")
parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker processes to use")
parser.add_argument("--chunksize", type=int, default=1, help="Chunk size for ProcessPoolExecutor instance")
cmd_args = parser.parse_args()
max_workers = cmd_args.max_workers
chunksize = cmd_args.chunksize

################################################################################
# Load the data
################################################################################

harp_data = pd.read_parquet("../harp_data/data/processed/hmi_sharp_cea_720s.parquet")
flare_data = pd.read_parquet("../flare_data/flare_data.parquet")

################################################################################
# Match flares to HARPs
################################################################################

flare_data = flare_data[["noaa_ar_5min", "start time", "peak time", "end time", "flare_class", "peak_intensity"]]
# For many flares, `noaa_ar_5min` is missing; these flares should be deleted before matching flares to HARPs because Pandas seems to match
# records if their join keys are both missing
flare_data = flare_data[~flare_data["noaa_ar_5min"].isna()]
# Many flares are missing end times; see notebooks/process_flare_data.ipynb for details
flare_data = flare_data[~flare_data["end time"].isna()]

max_times = harp_data.groupby("HARPNUM")["T_REC"].max().reset_index(name="max_time")
usable_harps = max_times.loc[max_times["max_time"] <= flare_data["end time"].max(), "HARPNUM"]
harp_data = harp_data[harp_data["HARPNUM"].isin(usable_harps)]

def match_flares_to_harp(one_harp_df: pd.DataFrame) -> pd.DataFrame:
    one_harp_df = pd.merge(one_harp_df, flare_data, how="left", left_on="NOAA_AR", right_on="noaa_ar_5min").drop(columns="noaa_ar_5min")

    is_match_invalid = (one_harp_df["T_REC"] < one_harp_df["start time"]) | (one_harp_df["T_REC"] > one_harp_df["end time"])
    one_harp_df.loc[is_match_invalid, ["start time", "peak time", "end time", "flare_class"]] = pd.NA
    one_harp_df.loc[is_match_invalid, "peak_intensity"] = np.nan
    one_harp_df.drop_duplicates(inplace=True)

    rec_counts = one_harp_df.groupby(["HARPNUM", "T_REC"]).size().reset_index(name="num_recs")
    one_harp_df = pd.merge(one_harp_df, rec_counts, how="inner", on=["HARPNUM", "T_REC"])
    should_keep = (one_harp_df["num_recs"] == 1) | ((one_harp_df["num_recs"] > 1) & ~one_harp_df["start time"].isna())
    one_harp_df = one_harp_df[should_keep].drop(columns="num_recs")

    return one_harp_df

one_harp_dfs = [one_harp_df for _, one_harp_df in harp_data.groupby("HARPNUM")]
with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("fork")) as executor:
    one_harp_dfs = list(executor.map(match_flares_to_harp, one_harp_dfs, chunksize=chunksize))
harp_flare_data = pd.concat(one_harp_dfs, ignore_index=True)

harp_flare_data.to_parquet("../combined_data/harp_flare_data.parquet")

script_elapsed_time = time.time() - script_start_time
print(f"\rMatching flares to HARPs ({int(script_elapsed_time)}s)", flush=True)
print("Done")
