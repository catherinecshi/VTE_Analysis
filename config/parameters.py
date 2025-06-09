"""
All thresholds, statistical settings, and processing parameters
"""

# DLC Filtering Parameters - from your dlc_processing.py filter_dataframe function
class DLCFilteringParams:
    likelihood_threshold = 0.95
    tracking_part = "greenLED"
    std_multiplier = 7  # for defining jump thresholds
    eps = 70  # DBSCAN maximum distance parameter
    min_samples = 40  # DBSCAN minimum samples parameter
    max_interpolation_distance = 100
    jump_threshold = 50
    smoothing_span = 3  # for smooth_points function

# VTE Analysis Parameters - from your trajectory analysis scripts
class VTEAnalysisParams:
    vte_threshold_multiplier = 1.5  # standard deviations above mean for VTE classification
    min_trajectory_length = 5  # minimum points in trajectory
    min_trajectory_time = 0.3  # minimum time in seconds
    max_trajectory_time = 4.0  # maximum time in seconds
    movement_threshold = 10  # threshold for detecting if rat is just staying in place

# Performance Analysis Parameters - from your performance_analysis.py
class PerformanceParams:
    criteria_threshold = 0.75  # 75% performance threshold
    min_trials_per_session = 5  # minimum trials needed for analysis
    
# Statistical Analysis Parameters - from your various analysis scripts
class StatisticalParams:
    alpha_level = 0.05
    bonferroni_correction = True
    min_observations_per_group = 5

# Plotting Parameters - to standardize your plots across scripts
class PlottingParams:
    figure_size = (10, 6)
    title_fontsize = 30
    label_fontsize = 24
    tick_fontsize = 20
    legend_fontsize = 24
    dpi = 300  # for high-quality saved figures