from src import data_processing
import numpy as np

save_path_2 = '/Users/catpillow/Documents/VTE_Analysis/data/VTE_Draft_2/BP11/inferenceTraining/Day1/old_dlc_2.csv'
save_path_1 = '/Users/catpillow/Documents/VTE_Analysis/data/VTE_Draft_2/BP11/inferenceTraining/Day1/old_dlc_1.csv'

df, _ = data_processing.concat_dlc(save_path_1, save_path_2)

print(df)


