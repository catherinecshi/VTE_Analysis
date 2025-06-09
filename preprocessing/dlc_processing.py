"""
main script for processing raw dlc data

creates
- coordinates.csv for each rat/day
    - x, y, and time values for each coordinate point
- filtered_points_info.csv for each rat
    - total number of points filtered out
    - # filtered from likelihood
    - # filtered from the start
    - # filtered from DBSCAN clustering
    - # filtered from jumps
"""

import os
import gc
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import DBSCAN

from config import settings
from config.paths import paths
from config.parameters import DLCFilteringParams
from preprocessing import data_processing
from utilities import time_utils
from utilities import logging_utils
from utilities import check_data
from visualization import generic_plots

# pylint: disable=logging-fstring-interpolation, unbalanced-tuple-unpacking, broad-exception-caught
logger = logging_utils.setup_script_logger()

def filter_dataframe(df: pd.DataFrame,
                     likelihood_threshold: float = DLCFilteringParams.likelihood_threshold,
                     tracking: str = DLCFilteringParams.tracking_part,
                     std_multiplier: int = DLCFilteringParams.std_multiplier,
                     eps: int = DLCFilteringParams.eps,
                     min_samples: int = DLCFilteringParams.min_samples,
                     max_interpolation_distance: int = DLCFilteringParams.max_interpolation_distance,
                     jump_threshold: int = DLCFilteringParams.jump_threshold,
                     start_index: Optional[int] = None): # currently keeps original indices
    """
    Filters dataframes + Check to make sure it's working properly
    Generally, >100 filtered out points is bad
    Keeps the original indices of the DataFrame
    
    All numbers referenced here refer to pixels
    Parameters:
        - df:               dataframe to be filtered
        - likelihood_threshold:     likelihood above which values would be included
                                    defaults to 0.95
        - track_part:       body part of rat to be used for their position
                            defaults to 'greenLED', which only exists for implanted rats
        - std_multiplier:   multiplier for std to define threshold beyond which jumps are excluded
                            defaults to 7
        - eps:              maximum distance between two samples for one to be considered as in the 
                            neighbourhood of another for DBCSCAN
                            defaults to 70
        - min_samples:      number of samples in a neighbourhood for a point to be 
                            considered a core point for DBSCAN
                            defaults to 40
        - max_interpolation_distance:   max distance between two consecutive points for interpolation
                                        to still happen
                                        defaults to 100
        - jump_threshold:   distance threshold above which something is considered a jump
                            defaults to 50
        - start_index:      index in df from which to start filtering
                            defaults to None (start from the first index)

    Returns:
        x & y : panda.Series : filtered and interpolated coordinates for x and y
    
    Procedure:
    1. filters based on the likelihood values
    2. filters out points before start_index if provided
    3. DBSCAN
    4. filters out based on std thresholds
    5. filters based on jumps
    6. interpolate
    
    Notes:
    - I make copies of df throughout so I can check how many points are filtered out at each step
    """
    
    # modify a copy instead of the original
    # 1. filter based on likelihood values
    likely_data = df[df[(tracking, "likelihood")] > likelihood_threshold].copy()
    unlikely_data = len(df) - len(likely_data)
    
    # 2. filter out points before the rat has started its first trial
    if start_index:
        likely_data = likely_data[likely_data.index >= start_index]
    
    if likely_data.empty:
        logger.error(f"{settings.CURRENT_RAT} on {settings.CURRENT_DAY} empty after filtering by likelihood")
        return None, None, None, None
    
    # 3. DBSCAN Cluster analysis
    coordinates = likely_data[[tracking]].copy()[[(tracking, "x"), (tracking, "y")]]
    coordinates.dropna(inplace = True) # don't drop nan for dbscan
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_
    noise_points_count = (labels == -1).sum() # so ik how many points were filtered out

    filtered_indices = labels != -1 # filter out noise
    filtered_data = likely_data[filtered_indices].copy()
    
    # 4. calculate thresholds
    diff_x = df[(tracking, "x")].diff().abs()
    diff_y = df[(tracking, "y")].diff().abs()
    threshold_x = diff_x.std() * std_multiplier
    threshold_y = diff_y.std() * std_multiplier
    
    # 5. calculate diff between current point and last non-jump point
    last_valid_index = 0
    jump_indices = [] # just to see how many points are jumped over
    
    for i in range(1, len(filtered_data)): # go through each consecutive pair of points
        diff_x = abs(filtered_data.iloc[i][(tracking, "x")] - filtered_data.iloc[last_valid_index][(tracking, "x")])
        diff_y = abs(filtered_data.iloc[i][(tracking, "y")] - filtered_data.iloc[last_valid_index][(tracking, "y")])
        
        # check for jumps
        if diff_x > threshold_x or diff_y > threshold_y:
            # add into list to see if it should be marked as NaN
            jump_indices.append(i)
        else:
            # udpate last valid index
            last_valid_index = i
            
    # only exclude jumps if it doesn't cross the jump threshold
    if len(jump_indices) < jump_threshold:
        for jump_index in jump_indices:
            filtered_data.at[filtered_data.index[jump_index], (tracking, "x")] = np.nan
            filtered_data.at[filtered_data.index[jump_index], (tracking, "y")] = np.nan
    else:
        logger.debug(f"{settings.CURRENT_RAT} on {settings.CURRENT_DAY} too many jumps")
    
    # 6. interpolating
    for axis in ["x", "y"]:
        valid_points = filtered_data[filtered_data[(tracking, axis)].notna()]
        for i in range(len(valid_points) - 1):
            start_idx = valid_points.index[i]
            end_idx = valid_points.index[i + 1]
            if end_idx - start_idx > 1:  # There are NaNs between these points
                if abs(filtered_data.at[end_idx, (tracking, axis)] - \
                   filtered_data.at[start_idx, (tracking, axis)]) <= max_interpolation_distance:
                    filtered_data.loc[start_idx:end_idx, (tracking, axis)] = \
                    filtered_data.loc[start_idx:end_idx, (tracking, axis)].interpolate()

    # final coordinate points
    x_coords = filtered_data[(tracking, "x")]
    y_coords = filtered_data[(tracking, "y")]
    
    # check how many points have been filtered out
    total_filtered_out = len(df) - len(x_coords)
    points_filtered = {"total": total_filtered_out, "likelihood": unlikely_data, "start": start_index,
                       "DBSCAN": noise_points_count, "jumps": len(jump_indices)}
    
    try: # because sometimes it's empty for some reason
        timepoints = filtered_data[("time", "time")]
    except IndexError:
        logger.debug(f"{settings.CURRENT_RAT} {settings.CURRENT_DAY} times empty")
        return x_coords, y_coords, pd.Series(), points_filtered
    
    return x_coords, y_coords, timepoints, points_filtered

def smooth_points(points: pd.Series, span: int = DLCFilteringParams.smoothing_span):
    """
    smooths points through a box-car configuration

    Args:
        - points: x or y coordinates
        - span: size for inclusion in box-car. Defaults to 3.

    Returns:
        list: new x or y coordinates after smoothing
    """
    new_points = [] # return variable
    for i, point in enumerate(points):
        if i < span: # not enough values to smooth yet
            new_points.append(point)
            continue
            
        if i > len(points) - span: # at the very end so can't smooth
            new_points.append(point)
            continue
        
        averaged_across = []
        for j in range(i - span, i + span):
            averaged_across.append(points.iloc[j])
        
        new_points.append(np.mean(averaged_across))
    
    return new_points

def check_too_filtered(df: pd.DataFrame, coords_x: pd.Series, coords_y: pd.Series, tracking: str, save: Optional[str] = None):
    """
    plots scatter plot to see how many points have been filtered out

    Args:
        - df: original DLC dataframe, before filtering
        - coords_x
        - coords_y
        - tracking: body part being tracked
        - save: file path if saving is desired for plot. Defaults to None.
    """
    
    unlikely_data = df[df[(tracking, "likelihood")] < 0.95].copy()
    unlikely_x = unlikely_data[(tracking, "x")]
    unlikely_y = unlikely_data[(tracking, "y")]
    
    x_values = [coords_x, unlikely_x]
    y_values = [coords_y, unlikely_y]
    generic_plots.create_populational_scatter_plot(x_values, y_values, save=save)



if __name__ == "__main__":
    data_structure = data_processing.load_data_structure(paths.vte_data)
    for rat, day_group in data_structure.items():
        filtered_info = []
        
        # determine which part to track rats on
        if any(rat == rat_ID for rat_ID in settings.IMPLANTED_RATS):
            track_part = "greenLED"
        else:
            track_part = "nose"
        
        settings.update_rat(rat)
        logger.info(f"starting processing for {rat}")
        
        # make folder if it doesn't exist for rat already
        rat_folder = paths.cleaned_dlc / rat
        if not os.path.exists(rat_folder):
            os.mkdir(rat_folder)
        
        for day, data in day_group.items():
            DLC_df = data_structure[rat][day][settings.DLC]
            SS_log = data_structure[rat][day][settings.SS]
            timestamps = data_structure[rat][day][settings.TIMESTAMPS]
            
            settings.update_day(day)
            logger.info(f"starting processing for {day}")
            
            # skip if already made
            save_path_coords = rat_folder / f"{day}_coordinates.csv"
            if os.path.exists(save_path_coords):
                continue
            
            # skip if no DLC or empty DLC
            if DLC_df is None or DLC_df.empty:
                logger.info(f"no or empty DLC for {rat} on {day}")
                continue
            
            if timestamps is not None:
                timestamps = check_data.check_timestamps(DLC_df, timestamps) # initial check of everything
                try:
                    DLC_df[("time", "time")] = timestamps
                except ValueError:
                    logger.error(f"unequal dlc frames {len(DLC_df)} and timestmaps {len(timestamps)} for {rat} on {day}")
                    continue
            else:
                logger.error(f"missing timestamps for {rat} on {day}")
                continue
            
            if SS_log is not None:
                try:
                    trial_starts = time_utils.get_video_trial_starts(timestamps, SS_log)
                    first_trial_start = next(iter(trial_starts)) # get the first trial start time to pass into filtering
                except Exception as e:
                    logger.debug(f"{e} with SS for {rat} on {day}")
                    x, y, times, points_filtered_out = filter_dataframe(DLC_df, tracking=track_part)
                else:
                    x, y, times, points_filtered_out = filter_dataframe(DLC_df, tracking=track_part, start_index=first_trial_start)
            else:
                x, y, times, points_filtered_out = filter_dataframe(DLC_df, tracking=track_part)
            
            # check for NoneTypes
            if x is None or y is None:
                continue
            
            new_x = smooth_points(x)
            new_y = smooth_points(y)

            coords_df = pd.DataFrame({"x": new_x, "y": new_y, "times": times})
            coords_df.dropna()
            coords_df.reset_index(drop=True)
            coords_df.to_csv(save_path_coords, index=False)
            
            check_too_filtered(DLC_df, x, y, track_part, save=rat_folder)
            
            # save info about what got filtered out
            if points_filtered_out is not None:
                save_filter_info = {"rat": rat,
                                    "day": day,
                                    "total": points_filtered_out["total"],
                                    "likelihood": points_filtered_out["likelihood"],
                                    "start": points_filtered_out["start"],
                                    "DBSCAN": points_filtered_out["DBSCAN"],
                                    "jumps": points_filtered_out["jumps"]
                                    }
                filtered_info.append(save_filter_info)
                logger.info("saved filtered out points data")
            
            # garbage disposal
            del DLC_df, SS_log, timestamps, x, y, new_x, new_y, coords_df, save_filter_info, points_filtered_out
            gc.collect()

        # overall info about how much got filtered out for rat
        save_path_filter = rat_folder / f"{rat}_filtered_points_info.csv"
        filtered_info_df = pd.DataFrame(filtered_info)
        filtered_info_df.to_csv(save_path_filter)
        
        del filtered_info, filtered_info_df
        gc.collect()
        
        logger.info(f"{rat} dlc processed")