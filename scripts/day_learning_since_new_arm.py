from src import data_processing
from src import performance_analysis

# loading
SAVE_PATH = '/Users/catpillow/Documents/VTE_DATA/VTE_Data'
DATA_STRUCTURE = data_processing.load_data_structure(SAVE_PATH)

# get the dataframe with days until new arm appeared
days_since_new_arm = performance_analysis.get_days_since_new_arm(SAVE_PATH, DATA_STRUCTURE)

# get the change in performance day by day
all_rats_performances = performance_analysis.create_all_rats_performance(data_structure=DATA_STRUCTURE)
all_rats_perf_changes = performance_analysis.create_all_rats_perf_changes(all_rats_performances)

# sort by the number of days since new arm was added
sorted_by_days_since_new_arm = days_since_new_arm.sort_values(by='days_since_new_arm')

# loop through to find the corresopnding change in performance for the day
for index, row in sorted_by_days_since_new_arm.iterrows():
    day = row['day']
    rat = row['rat']
    
    # find the corresponding change in performance
    