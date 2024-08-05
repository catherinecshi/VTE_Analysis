import logging
import pandas as pd
import numpy as np

from src import data_processing
from src import performance_analysis
from src import plotting

logging.basicConfig(filename="day_learning_since_new_arm_log.txt",
                    format='%(asctime)s %(message)s',
                    filemode="w")

logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# loading
SAVE_PATH = "/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data"
DATA_STRUCTURE = data_processing.load_data_structure(SAVE_PATH)

# get the dataframe with days until new arm appeared
days_since_new_arm = performance_analysis.get_days_since_new_arm(SAVE_PATH, DATA_STRUCTURE)

# get the change in performance day by day
all_rats_performances = performance_analysis.create_all_rats_performance(data_structure=DATA_STRUCTURE)
all_perf_changes_by_trial = performance_analysis.create_all_perf_changes_by_trials(all_rats_performances)

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

# Merge SEM values back into the main dataframe
merged_df = pd.merge(learning_during_volatility_df, sem_by_day, on="days_since_new_arm", how="left")

plotting.create_box_and_whisker_plot(learning_during_volatility_df, x="days_since_new_arm", y="perf_change",
                                    title="Learning during Volatility",
                                    xlabel="Number of Days Since New Arm Added",
                                    ylabel="Change in Performance since Last Session")

plotting.create_histogram(learning_during_volatility_df, "days_since_new_arm", "perf_change",
                          title="Learning during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="Change in Performance since Last Session")

plotting.create_line_plot(learning_during_volatility_df["days_since_new_arm"],
                          learning_during_volatility_df["perf_change"],
                          yerr=merged_df["sem"],
                          title="Learning during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="Change in Performance since Last Session")