"""
plots the day by day performance change against days since new arm was introduced
day by day performance change goes as - 
if new arm was introduced on day 1, diff betweeen day 50% & day 1 is taken
"""

import logging
import pandas as pd
import numpy as np

from config.paths import paths
from preprocessing import data_processing
from analysis import performance_analysis
from visualization import generic_plots

logging.basicConfig(filename="day_learning_since_new_arm_log.txt",
                    format='%(asctime)s %(message)s',
                    filemode="w")

logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# loading
SAVE_PATH = paths.vte_data
DATA_STRUCTURE = data_processing.load_data_structure(SAVE_PATH)

# get the dataframe with days until new arm appeared
days_since_new_arm = performance_analysis.get_days_since_new_arm(SAVE_PATH, DATA_STRUCTURE)

# get the change in performance day by day
all_rats_performances = performance_analysis.get_all_rats_performance(data_structure=DATA_STRUCTURE)
all_perf_changes_by_trial = performance_analysis.save_all_perf_changes_trials(all_rats_performances)

# ensure type is the same
days_since_new_arm["trials_available"] = days_since_new_arm["trials_available"].apply(lambda x: [int (y) for y in x])
days_since_new_arm = days_since_new_arm.astype({"rat": "str",
                                                "day": "int", 
                                                "arm_added": "bool", 
                                                "days_since_new_arm": "int"})
all_perf_changes_by_trial = all_perf_changes_by_trial.astype({"rat": "str",
                                                              "day": "int", 
                                                              "trial_type": "int", 
                                                              "perf_change": "float"})

# create an array of just the corresponding trial type change
rat_df = days_since_new_arm.groupby("rat")
learning_during_volatility = []
for rat, rat_group in rat_df:
    sorted_by_day_df = rat_group.sort_values(by="day")
    for i, row in sorted_by_day_df.iterrows():
        day = row["day"]
        number_of_days_since = row["days_since_new_arm"]
        try:
            highest_trial_available = max(row["trials_available"])
        except ValueError as e:
            print(f"value error {e} with {rat} on {day}")
        corresponding_row = None

        corresponding_row = all_perf_changes_by_trial[(all_perf_changes_by_trial["rat"] == rat) & \
                                                      (all_perf_changes_by_trial["day"] == day) & \
                                                      (all_perf_changes_by_trial["trial_type"] == highest_trial_available)]
                                                      
        if corresponding_row.empty:
            print(f"error for {rat} on {day} - no corresponding perf change")
            continue

        corresponding_perf_change = corresponding_row["perf_change"].iloc[0]
        learning_during_volatility.append({"rat": rat, "day": day, "trial_type": highest_trial_available,
                                           "days_since_new_arm": number_of_days_since,
                                           "perf_change": corresponding_perf_change})

learning_during_volatility_df = pd.DataFrame(learning_during_volatility)
learning_during_volatility_df.to_csv("/Users/catpillow/Documents/VTE_Analysis/processed_data/learning_during_volatility.csv")

# Calculate SEM for each group of days_since_new_arm
def calculate_sem(data):
    n = len(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    return sem

sem_by_day = learning_during_volatility_df.groupby("days_since_new_arm")["perf_change"].apply(calculate_sem).reset_index()
sem_by_day.columns = ["days_since_new_arm", "sem"]

mean_by_day = learning_during_volatility_df.groupby("days_since_new_arm")["perf_change"].mean().reset_index()
mean_by_day.columns = ["days_since_new_arm", "mean_perf_change"]

# Merge SEM values back into the main dataframe
merged_df = pd.merge(mean_by_day, sem_by_day, on="days_since_new_arm", how="left")

"""plotting.create_box_and_whisker_plot(learning_during_volatility_df, x="days_since_new_arm", y="perf_change",
                                    xlim=15, title="Learning during Volatility",
                                    xlabel="Number of Days Since New Arm Added",
                                    ylabel="Change in Performance since Last Session")

plotting.create_histogram(learning_during_volatility_df, "days_since_new_arm", "perf_change",
                          title="Learning during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="Change in Performance since Last Session")"""

generic_plots.create_line_plot(merged_df["days_since_new_arm"],
                          merged_df["mean_perf_change"],
                          merged_df["sem"],
                          xlim=(0, 8),
                          ylim=(-10, 20),
                          title="Learning during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="Change in Performance\nsince Last Session (%)")