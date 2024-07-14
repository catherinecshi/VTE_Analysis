from src import data_processing
import numpy as np

save_path_2 = '/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data/BP08/inferenceTraining/Day8/BP08_20220715_inferenceTraining_Day8_2.1.videoTimeStamps.npy'
save_path_1 = '/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data/BP08/inferenceTraining/Day8/BP08_20220715_inferenceTraining_Day8.1.videoTimeStamps.npy'
timestamps_1 = np.load(save_path_1)
print(timestamps_1)
timestamps_2 = np.load(save_path_2)
print(timestamps_2)