import numpy as np
import pandas as pd
import time
import tracemalloc
from utils import *

tracemalloc.start()

print("Making plots...")
script_start_time = time.time()

section_start_time = time.time()

harp_flare_data = pd.read_parquet(
    "../combined_data/processed_high-qual_near-center-70_no-nas_flares.parquet"
)
harp_flare_data = harp_flare_data[
    ["HARPNUM", "T_REC"] + all_predictors + ["peak time", "flare_class"]
]
# It is shown in harp_data/hmi_sharp_cea_720s/data/impute_harp_data.ipynb that
# several predictors are strongly right-skewed. We log-transform those
# predictors. If we didn't, then their min-max-scaled values would largely be
# close to zero, and it would be hard to discern any patterns in those values.
right_skewed_predictors = [
    "TOTUSJH", "TOTUSJZ", "USFLUX", "TOTPOT", "ABSNJZH", "SAVNCPP", "NACR",
    "SIZE_ACR", "SIZE", "NPIX"
]
harp_flare_data[right_skewed_predictors] = (
    harp_flare_data[right_skewed_predictors]
    .apply(lambda x: np.log1p(x - x.min()))
)

section_elapsed_time = int(time.time() - section_start_time)
current_size, peak_size = tracemalloc.get_traced_memory()
current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
print("After reading, subsetting, log-transforming harp_flare_data")
print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
tracemalloc.reset_peak()

section_start_time = time.time()

minmax_data = harp_flare_data.copy()
minmax_data[all_predictors] = minmax_scale(minmax_data[all_predictors])

section_elapsed_time = int(time.time() - section_start_time)
current_size, peak_size = tracemalloc.get_traced_memory()
current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
print("After making minmax_data")
print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
tracemalloc.reset_peak()

section_start_time = time.time()

minmax_plots = (
    minmax_data
    .groupby("HARPNUM")[minmax_data.columns]
    .apply(make_harp_flare_plot, transform_type="minmax")
)

section_elapsed_time = int(time.time() - section_start_time)
current_size, peak_size = tracemalloc.get_traced_memory()
current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
print("After making minmax_plots")
print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
tracemalloc.reset_peak()

section_start_time = time.time()

rank_data = harp_flare_data.copy()
rank_data[all_predictors] = rank_data[all_predictors].apply(rank_transform)

section_elapsed_time = int(time.time() - section_start_time)
current_size, peak_size = tracemalloc.get_traced_memory()
current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
print("After making rank_data")
print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
tracemalloc.reset_peak()

section_start_time = time.time()

rank_plots = (
    rank_data
    .groupby("HARPNUM")[rank_data.columns]
    .apply(make_harp_flare_plot, transform_type="rank")
)

section_elapsed_time = int(time.time() - section_start_time)
current_size, peak_size = tracemalloc.get_traced_memory()
current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
print("After making rank_plots")
print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
tracemalloc.reset_peak()

section_start_time = time.time()

harp_by_harp_minmax_plots = (
    harp_flare_data
    .groupby("HARPNUM")[harp_flare_data.columns]
    .apply(
        make_harp_flare_plot, transform_type="minmax", need_to_transform=True
    )
)

section_elapsed_time = int(time.time() - section_start_time)
current_size, peak_size = tracemalloc.get_traced_memory()
current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
print("After making harp_by_harp_minmax_plots")
print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
tracemalloc.reset_peak()

section_start_time = time.time()

harp_by_harp_rank_plots = (
    harp_flare_data
    .groupby("HARPNUM")[harp_flare_data.columns]
    .apply(make_harp_flare_plot, transform_type="rank", need_to_transform=True)
)

section_elapsed_time = int(time.time() - section_start_time)
current_size, peak_size = tracemalloc.get_traced_memory()
current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
print("After making harp_by_harp_rank_plots")
print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
tracemalloc.reset_peak()

section_start_time = time.time()

flare_class_flags = (
    harp_flare_data
    .groupby("HARPNUM")["flare_class"]
    .apply(calc_flare_class_flags)
    .reset_index(level=1, drop=True)
)

section_elapsed_time = int(time.time() - section_start_time)
current_size, peak_size = tracemalloc.get_traced_memory()
current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
print("After making flare_class_flags")
print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
tracemalloc.reset_peak()

for flare_class in ["m_plus", "x"]:
    section_start_time = time.time()

    save_as_pdf_pages(
        minmax_plots[flare_class_flags[f"{flare_class}_flag"]],
        filename=f"all_harps_scaling/minmax/{flare_class}_plots.pdf"
    )

    section_elapsed_time = int(time.time() - section_start_time)
    current_size, peak_size = tracemalloc.get_traced_memory()
    current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
    print(f"After saving all_harps_scaling/minmax/{flare_class}_plots.pdf")
    print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
    tracemalloc.reset_peak()

    section_start_time = time.time()

    save_as_pdf_pages(
        rank_plots[flare_class_flags[f"{flare_class}_flag"]],
        filename=f"all_harps_scaling/rank/{flare_class}_plots.pdf"
    )

    section_elapsed_time = int(time.time() - section_start_time)
    current_size, peak_size = tracemalloc.get_traced_memory()
    current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
    print(f"After saving all_harps_scaling/rank/{flare_class}_plots.pdf")
    print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
    tracemalloc.reset_peak()

    section_start_time = time.time()

    save_as_pdf_pages(
        harp_by_harp_minmax_plots[flare_class_flags[f"{flare_class}_flag"]],
        filename=f"harp_by_harp_scaling/minmax/{flare_class}_plots.pdf"
    )

    section_elapsed_time = int(time.time() - section_start_time)
    current_size, peak_size = tracemalloc.get_traced_memory()
    current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
    print(f"After saving harp_by_harp_scaling/minmax/{flare_class}_plots.pdf")
    print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
    tracemalloc.reset_peak()

    section_start_time = time.time()

    save_as_pdf_pages(
        harp_by_harp_rank_plots[flare_class_flags[f"{flare_class}_flag"]],
        filename=f"harp_by_harp_scaling/rank/{flare_class}_plots.pdf"
    )

    section_elapsed_time = int(time.time() - section_start_time)
    current_size, peak_size = tracemalloc.get_traced_memory()
    current_size, peak_size = current_size / 2 ** 30, peak_size / 2 ** 30
    print(f"After saving harp_by_harp_scaling/rank/{flare_class}_plots.pdf")
    print(f"{current_size=}, {peak_size=}, {section_elapsed_time=}")
    tracemalloc.reset_peak()

script_elapsed_time = time.time() - script_start_time
print(f"Done ({int(script_elapsed_time)}s)")
