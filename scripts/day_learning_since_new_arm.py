import logging
import pandas as pd

from src import data_processing
from src import performance_analysis
from src import plotting

logging.basicConfig(filename='day_learning_since_new_arm_log.txt',
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# loading
SAVE_PATH = '/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data'
DATA_STRUCTURE = data_processing.load_data_structure(SAVE_PATH)

# get the dataframe with days until new arm appeared
days_since_new_arm = performance_analysis.get_days_since_new_arm(SAVE_PATH, DATA_STRUCTURE)

# get the change in performance day by day
all_rats_performances = performance_analysis.create_all_rats_performance(data_structure=DATA_STRUCTURE)
all_rats_perf_changes = performance_analysis.create_all_perf_changes(all_rats_performances)

# dataframe for overall change in perf
merged_df = pd.merge(all_rats_perf_changes, days_since_new_arm, how="inner", on=["rat", "day"])

plotting.create_box_and_whisker_plot(merged_df, x="days_since_new_arm", y="perf_change",
                                    title="Learning during Volatility",
                                    xlabel="Number of Days Since New Arm Added",
                                    ylabel="Change in Performance since Last Session")

plotting.create_histogram(merged_df, "days_since_new_arm", "perf_change",
                          title="Learning during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="Change in Performance since Last Session")
