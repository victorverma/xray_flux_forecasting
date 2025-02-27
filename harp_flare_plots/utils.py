import pandas as pd
from plotnine import *
from sklearn.preprocessing import minmax_scale

# The predictor groups from Figure 6 of https://doi.org/10.1029/2019SW002214. An
# LSTM was trained to predict whether a flare is strong (M- or X-class) or weak
# (B-class) using 24 hours of predictor data ending six hours before the flare.
# For each group, a model was trained with just the predictors in that group;
# also, for each predictor, a model was trained with just that predictor. In the
# list below, the groups are sorted in descending order by test accuracy. Within
# each group, the predictors are sorted in descending order by test accuracy.
# 
# 1. TOTUSJH TOTUSJZ USFLUX TOTPOT
# 2. ABSNJZH SAVNCPP
# 3. NACR SIZE_ACR SIZE NPIX
# 4. MEANPOT MEANJZH MEANSHR SHRGT45 MEANALP MEANJZD MEANGBT MEANGAM MEANGBZ MEANGBH
#
# The groups and predictors are in reverse order below so that they will be
# properly sorted in plots.
all_predictors = [
    "MEANGBH", "MEANGBZ", "MEANGAM", "MEANGBT", "MEANJZD", "MEANALP", "SHRGT45", "MEANSHR", "MEANJZH", "MEANPOT",
    "NPIX", "SIZE", "SIZE_ACR", "NACR",
    "SAVNCPP", "ABSNJZH",
    "TOTPOT", "USFLUX", "TOTUSJZ", "TOTUSJH"
]
sharp_params = all_predictors[:10] + all_predictors[14:]

def rank_transform(x: pd.Series) -> pd.Series:
    """
    Compute the ranks of the numbers in a Series and then affinely transform
    them so that rank one is mapped to zero and the biggest rank is mapped to
    one.

    :param x: A Series of numbers
    :return: A Series with the affinely-transformed ranks
    """
    r = x.rank(method="dense")
    r = (r - 1) / (r.max() - 1)
    return r

def make_time_label(time: pd.Timestamp) -> str:
    """
    Create a formatted time label from a Pandas Timestamp, including the date, 
    time, and time zone abbreviation.

    :param time: The Timestamp to create a label from
    :return: A label with the format "YYYY-MM-DD\nHH:MM:SS TZ"
    """
    tz_abbr = str(time.tzinfo)
    label = time.strftime(f'%Y-%m-%d\n%H:%M:%S {tz_abbr}')
    return label

def make_fill_caption_title(
        transform_type: str,
        num_flares: int,
        need_to_transform: bool,
        harp_num: int
    ) -> tuple[str, str, str]:
    """
    Make the fill, caption, and title for the plot make_harp_flare_plot makes.

    :param transform_type: The type of the transform that was or should be
        applied to the predictors ("minmax" or "rank")
    :param num_flares: The number of flares produced by the HARP of interest
    :param need_to_transform: Whether to transform the predictors
    :param harp_num: The number of the HARP of interest
    :return: A tuple containing the fill, caption, and title
    """
    if transform_type == "minmax":
        fill = "Min-Max-Scaled\nValue"
    elif transform_type == "rank":
        fill = "Rank-Transformed\nValue"

    if num_flares > 0:
        caption_prefix = "Vertical lines are at flare peak times."
    else:
        caption_prefix = "There were no C-, M-, or X-class flares."
    if need_to_transform:
        caption_middle = "For each predictor, scaling used values from this HARP only."
    else:
        caption_middle = "For each predictor, scaling used values from all HARPs."
    caption_suffix = "Predictors are grouped and sorted according to Figure 6 of https://doi.org/10.1029/2019SW002214."
    caption = "\n".join([caption_prefix, caption_middle, caption_suffix])

    title = f"HARP {harp_num}"

    return fill, caption, title

def make_harp_flare_plot(
        one_harp_data: pd.DataFrame,
        transform_type: str,
        need_to_transform: bool = False,
        use_sharp_params_only: bool = False
    ) -> ggplot:
    """
    Make a plot that shows, for a given HARP, how various predictors, like the
    SHARP parameters, evolved over time, as well as the flares that the HARP
    produced.

    :param one_harp_data: A DataFrame with predictor data for one HARP; it
        should be a subset of a DataFrame produced using
        data/match_flares_to_harps.py 
    :param transform_type: The type of the transform that was or should be
        applied to the predictors ("minmax" or "rank")
    :param need_to_transform: Whether to transform the predictors
    :param use_sharp_params_only: Whether the SHARP parameters should be the
        only predictors
    :return: The plot
    """
    transforms = {"minmax": minmax_scale, "rank": rank_transform}
    if transform_type not in transforms:
        raise ValueError(f"transform_type must be one of {transforms.keys()}.")
    else:
        transform = transforms[transform_type]

    harp_num = one_harp_data["HARPNUM"].iloc[0]
    first_time = one_harp_data["T_REC"].iloc[0]
    first_time_label = make_time_label(first_time)
    last_time = one_harp_data["T_REC"].iloc[-1]
    last_time_label = make_time_label(last_time)

    flare_classes = ["C", "M", "X"]
    flares = one_harp_data.loc[
        one_harp_data["peak time"].notna() \
        & one_harp_data["flare_class"].isin(flare_classes),
        ["peak time", "flare_class"]
    ].drop_duplicates()
    flares["flare_class"] = pd.Categorical(
        flares["flare_class"], categories=flare_classes
    )

    predictors = sharp_params if use_sharp_params_only else all_predictors
    one_harp_data = one_harp_data[["T_REC"] + predictors].copy()
    if need_to_transform:
        one_harp_data[predictors] = transform(one_harp_data[predictors])

    long_one_harp_data = pd.melt(
        one_harp_data, id_vars="T_REC", var_name="predictor", value_name="val"
    )
    long_one_harp_data["predictor"] = long_one_harp_data["predictor"].map({
        predictor: i + 1 for i, predictor in enumerate(predictors)
    })

    x_breaks=[first_time, last_time]
    x_labels=[first_time_label, last_time_label]
    y_breaks = list(range(1, len(predictors) + 1))
    # For drawing horizontal lines that separate the predictor groups
    y_intercepts = [10.5, 12.5] if use_sharp_params_only else [10.5, 14.5, 16.5]

    fill, caption, title = make_fill_caption_title(
        transform_type, len(flares), need_to_transform, harp_num
    )

    plot = (
        ggplot(long_one_harp_data, aes(x="T_REC", y="predictor", fill="val")) +
        geom_tile() +
        geom_hline(
            yintercept=y_intercepts, color="green", linetype="solid", size=1
        ) +
        geom_vline(
            aes(xintercept="peak time", color="flare_class"),
            flares,
            linetype="dashed"
        ) +
        scale_x_datetime(breaks=x_breaks, labels=x_labels) +
        scale_y_continuous(
            breaks=y_breaks, labels=predictors, minor_breaks=y_intercepts
        ) +
        scale_fill_gradient(low="white", high="black", limits=(0, 1)) +
        scale_color_manual(values=["blue", "orange", "red"]) +
        labs(
            x="Time", y="Predictor", color="Flare\nClass", fill=fill,
            title=title, caption=caption
        ) +
        theme_classic() +
        theme(
            axis_title_y=element_text(margin={"r": -20}),
            axis_ticks_minor_y=element_line(color="green", size=1.7),
            axis_ticks_length_minor_y=50,
            legend_position="top"
        )
    )

    return plot

def calc_flare_class_flags(flare_classes: pd.Series) -> pd.DataFrame:
    """
    Given a Series whose values are flare classes ("A", "B", "C", "M", "X") or
    pd.NA's, construct a DataFrame with two columns that indicate whether there
    is at least one M+ class and whether there is at least one X class.

    :param flare_classes: A Series whose values are flare classes or pd.NA's
    :return: A DataFrame whose columns contain the flare class flags
    """
    m_plus_flag = flare_classes.isin(["M", "X"]).any()
    x_flag = (flare_classes == "X").any()
    flare_class_flags = pd.DataFrame({
        "m_plus_flag": [m_plus_flag], "x_flag": [x_flag]
    })
    return flare_class_flags
