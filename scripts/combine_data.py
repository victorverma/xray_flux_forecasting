import argparse
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

print("Combining data", end="", flush=True)
script_start_time = time.time()

################################################################################
# Parse the command-line arguments
################################################################################

parser = argparse.ArgumentParser(description="Combine HARP, flux, and flare data in one data frame")
parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker processes to use")
parser.add_argument("--chunksize", type=int, default=1, help="Chunk size for ProcessPoolExecutor instance")
cmd_args = parser.parse_args()
max_workers = cmd_args.max_workers
chunksize = cmd_args.chunksize

################################################################################
# Load the data
################################################################################

harp_data = pd.read_parquet("../harp_data/data/processed/aggregated_high-qual_near-center-70.parquet")
flux_data = pd.read_parquet("../xray_fluxes/data/processed/1m_data.parquet")
flare_data = pd.read_csv("../flare_data/sci_20100101_20240721.csv")

################################################################################
# Fix the issues in the flare list
################################################################################

flare_data["start time"] = pd.to_datetime(flare_data["start time"], errors="coerce", format="%Y/%m/%d %H:%M", utc=True)
flare_data["end time"] = pd.to_datetime(flare_data["end time"], errors="coerce", format="%Y/%m/%d %H:%M", utc=True)
flare_data["peak time"] = pd.to_datetime(flare_data["peak time"], errors="coerce", format="%Y/%m/%d %H:%M", utc=True)

def get_peak_intensity(fl_class: str) -> float:
    flare_class = fl_class[0]
    multiplier = float(fl_class[1:])
    powers = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
    return multiplier * powers[flare_class]

flare_data["peak_intensity"] = flare_data["fl_class"].map(get_peak_intensity)

def get_flare_class(peak_intensity: float) -> str:
    thresholds = [10 ** i for i in range(-4, -9, -1)]
    flare_classes = ["X", "M", "C", "B", "A"]
    for threshold, flare_class in zip(thresholds, flare_classes):
        if peak_intensity >= threshold:
            return flare_class
    return pd.NA

flare_data["flare_class"] = flare_data["peak_intensity"].map(get_flare_class)

################################################################################
# Combine all the data
################################################################################

combined_data = pd.merge(harp_data, flux_data[["time", "flux"]], how="inner", left_on="T_REC", right_on="time").drop(columns="time")
window_ends = combined_data["T_REC"] + pd.Timedelta(hours=23, minutes=59)
should_keep = (window_ends <= flux_data["time"].iloc[-1]) & (window_ends <= flare_data["peak time"].iloc[-1])
combined_data = combined_data[should_keep]

def compute_maxes(t):
    window_start = t
    window_end = t + pd.Timedelta(hours=23, minutes=59)

    is_in_window = (flux_data["time"] >= window_start) & (flux_data["time"] <= window_end)
    max_flux = flux_data.loc[is_in_window, "flux"].max()

    is_in_window = (flare_data["peak time"] >= window_start) & (flare_data["peak time"] <= window_end)
    if is_in_window.any():
        flare_classes = flare_data.loc[is_in_window, "flare_class"]
        peak_intensities = flare_data.loc[is_in_window, "peak_intensity"]
        i = peak_intensities.idxmax()
        max_flare_class = flare_classes[i]
        max_peak_intensity = peak_intensities[i]
    else:
        max_flare_class = pd.NA
        max_peak_intensity = np.nan

    return max_flux, max_flare_class, max_peak_intensity

max_tuples = []
with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("fork")) as executor:
    max_tuples = list(executor.map(compute_maxes, combined_data["T_REC"], chunksize=chunksize))
max_fluxes, max_flare_classes, max_peak_intensities = zip(*max_tuples)
combined_data["max_flux_next_24h"] = max_fluxes
combined_data["max_flare_class_next_24h"] = max_flare_classes
# When workers return results, pd.NA is changed to None during serialization. The string data type is a nullable Pandas data type; the type
# change below causes None to become pd.NA. Also, the type change will prevent pd.NA from being changed to None in the parquet file.
combined_data["max_flare_class_next_24h"] = combined_data["max_flare_class_next_24h"].astype("string")
combined_data["max_peak_intensity_next_24h"] = max_peak_intensities

combined_data.to_parquet("../combined_data/combined_data.parquet")

script_elapsed_time = time.time() - script_start_time
print(f"\rCombining data ({int(script_elapsed_time)}s)", flush=True)
print("Done")
