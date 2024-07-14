from src import data_processing

SAVE_PATH = '/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data'
#data_processing.convert_all_timestamps(SAVE_PATH)
#data_processing.convert_all_statescripts(SAVE_PATH)
data_processing.concat_duplicates(SAVE_PATH)