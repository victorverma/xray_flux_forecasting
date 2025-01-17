import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

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
should_keep = combined_data["T_REC"] + pd.Timedelta(hours=23, minutes=59) <= flux_data["time"].iloc[-1]
combined_data = combined_data[should_keep]

max_fluxes = []
max_flare_classes = []
max_peak_intensities = []
for t in combined_data["T_REC"]:
    window_start = t
    window_end = t + pd.Timedelta(hours=23, minutes=59)

    is_in_window = (flux_data["time"] >= window_start) & (flux_data["time"] <= window_end)
    max_flux = flux_data.loc[is_in_window, "flux"].max()
    max_fluxes.append(max_flux)

    is_in_window = (flare_data["peak time"] >= window_start) & (flare_data["peak time"] <= window_end)
    flare_classes = flare_data.loc[is_in_window, "flare_class"]
    peak_intensities = flare_data.loc[is_in_window, "peak_intensity"]
    i = peak_intensities.idxmax()
    max_flare_classes.append(flare_classes[i])
    max_peak_intensities.append(peak_intensities[i])
combined_data["max_flux_next_24h"] = max_fluxes
combined_data["max_flare_class_next_24h"] = max_flare_classes
combined_data["max_peak_intensity_next_24h"] = max_peak_intensities

combined_data.to_parquet("../combined_data/combined_data.parquet")
