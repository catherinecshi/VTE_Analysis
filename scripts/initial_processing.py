from src import data_processing

SAVE_PATH = "/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data"
BP07_PATH = "/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data/BP07/inferenceTraining"
data_processing.initial_to_inference(BP07_PATH)
data_processing.convert_all_timestamps(SAVE_PATH)
data_processing.convert_all_statescripts(SAVE_PATH)
data_processing.concat_duplicates(SAVE_PATH)