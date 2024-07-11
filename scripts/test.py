from src import data_processing

save_path = '/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data'
data_processing.convert_all_timestamps(save_path)
data_processing.concat_duplicates(save_path)