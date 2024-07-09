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
logging.debug(all_rats_performances)
all_perf_changes_by_trial = performance_analysis.create_all_perf_changes_by_trials(all_rats_performances)
logging.debug(all_perf_changes_by_trial)

# create an array of just the corresponding trial type change
trials_available_df = days_since_new_arm.groupby('trials_available')

for trials_available, group in trials_available_df:
    

# sort by the number of days since new arm was added
sorted_by_days_since_new_arm = days_since_new_arm.sort_values(by='days_since_new_arm')



plotting.create_box_and_whisker_plot(merged_df, x='days_since_new_arm', y='perf_change',
                                    title='Learning during Volatility',
                                    xlabel='Number of Days Since New Arm Added',
                                    ylabel='Change in Performance since Last Session')

plotting.create_histogram(merged_df, 'days_since_new_arm', 'perf_change',
                          title='Learning during Volatility',
                          xlabel='Number of Days Since New Arm Added',
                          ylabel='Change in Performance since Last Session')



#scrap
"""
current_number_days = 0
learning_vs_volatility = []
learning_vs_volatility.append([])
for index_i, row_i in sorted_by_days_since_new_arm.iterrows():
    # find the corresponding change in performance
    for index_j, row_j in all_rats_perf_changes.iterrows():
        if not(row_i['day'] is row_j['day'] and row_i['rat'] is row_j['rat']): # if not match between rat & day
            continue

        trial_type = row_j['trial_type']
        available_arms = row_i['trials_available']
        
        # check that the trial types match up
        int_trial_type = helper_functions.string_to_int_trial_types(trial_type)
        if int_trial_type not in available_arms:
            logging.error(f'incongruence between {available_arms} and {int_trial_type}'
                            f' for {row_i['rat']} on {row_i['day']}')

        # get the most recently added trial type
        most_recent_trial = max(available_arms)
        match = helper_functions.trial_type_equivalency(most_recent_trial, trial_type)
        
        if not match: # skip if not most recent trial type
            continue
            
        if row_i['days_since_new_arm'] is current_number_days:
            learning_vs_volatility[current_number_days].append(row_j['perf_change'])
        else:
            current_number_days = row_i['days_since_new_arm']
            learning_vs_volatility.append([])
            learning_vs_volatility[current_number_days].append(row_j['perf_change'])
"""
