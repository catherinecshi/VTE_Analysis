import os
import pickle
import pandas as pd
import numpy as np

import data_processing
import calculating_VTEs
import performance_analysis

# creating the main data structure
#base_path = '/Users/catpillow/Documents/VTE Analysis/Data_draft'
#main_data_structure = data_structure.create_main_data_structure(base_path)

# saving
#save_path = '/Users/catpillow/Documents/VTE Analysis/VTE_Data' # this is just SS (added BP13 DLC & timestamps)
#save_path_BP = '/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13'
#save_path = '/Users/catpillow/Downloads/VTE_Data'

# loading
#loaded_data_structure = data_structure.load_data_structure(save_path)
#BP13_data = data_processing.load_one_rat(save_path_BP)

# example
ratID = 'BP13'
day = 'Day8'

#zIdPhi, IdPhi, trajectories = calculating_VTEs.quantify_VTE(BP13_data, ratID, day, save = False)
#rat_VTE_over_sessions(BP13_data, ratID)

#rat_VTE_over_sessions(loaded_data_structure, ratID)
