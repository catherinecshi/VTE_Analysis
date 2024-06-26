import os
import pickle
import pandas as pd
import numpy as np

import data_processing
import calculating_VTEs
import performance_analysis

# creating the main data structure
base_path = '/Users/catpillow/Documents/VTE Analysis/Data_Draft'
main_data_structure = data_processing.create_main_data_structure(base_path)

# saving
save_path = '/Users/catpillow/Documents/VTE Analysis/VTE_Data'
data_processing.save_data_structure(main_data_structure, save_path)

# loading
loaded_data_structure = data_processing.load_data_structure(save_path)

# example
ratID = 'BP13'
day = 'Day8'

#zIdPhi, IdPhi, trajectories = calculating_VTEs.quantify_VTE(BP13_data, ratID, day, save = False)
#rat_VTE_over_sessions(BP13_data, ratID)

#rat_VTE_over_sessions(loaded_data_structure, ratID)
