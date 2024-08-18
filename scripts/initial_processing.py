from src import data_processing
from src import readCameraModuleTimeStamps
import numpy as np

SAVE_PATH = "/Users/catpillow/Documents/VTE_Analysis/data/timestamps"
BP07_PATH = "/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data/BP07/inferenceTraining"
#data_processing.initial_to_inference(BP07_PATH)
data_processing.convert_all_timestamps(SAVE_PATH)
#data_processing.convert_all_statescripts(SAVE_PATH)
#data_processing.concat_duplicates(SAVE_PATH)

"""file_path_2 = "/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data/BP19/inferenceTraining/Day3/BP19_20230912_inferenceTraining_Day3.1.videoTimeStamps.npy"
file_path = "/Users/catpillow/Downloads/BP19_20230912_inferenceTraining_Day3.1.videoTimeStamps"
timestamps = readCameraModuleTimeStamps(file_path)
timestamps_2 = np.load(file_path_2)
print(timestamps)
print(timestamps_2)"""

