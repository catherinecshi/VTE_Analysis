import os
import pandas as pd

from src import helper
from src import plotting

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")

IdPhis = []
choice_IdPhis = {}
trial_type_IdPhis = {}
for rat in os.listdir(base_path):
    rat_path = os.path.join(base_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file)
            zIdPhi_csv = pd.read_csv(file_path)
            
            for index, row in zIdPhi_csv.iterrows():
                IdPhi = row["IdPhi"]
                choice = row["Choice"]
                trial_type = row["Trial_Type"]
                
                IdPhis.append(IdPhi)
                
                # check if key exists before adding to dictionary
                if choice in choice_IdPhis.keys():
                    choice_IdPhis[choice].append(IdPhi)
                else:
                    choice_IdPhis[choice] = []
                    choice_IdPhis[choice].append(IdPhi)
                
                if trial_type not in trial_type_IdPhis.keys():
                    trial_type_IdPhis[trial_type] = []
                trial_type_IdPhis[trial_type].append(IdPhi)

# make figure for counts of durations
plotting.create_frequency_histogram(IdPhis, title="IdPhi Distribution", stat="count", xlim=(0, 150),
                                    binwidth=2, xlabel="IdPhi", ylabel="Counts")
plotting.create_multiple_frequency_histograms(choice_IdPhis, title="IdPhi Distribution by Choice Arm",
                                              binwidth=5, xlabel="IdPhi", ylabel="Density")
plotting.create_multiple_frequency_histograms(trial_type_IdPhis, title="IdPhi Distribution by Trial Type",
                                              binwidth=5, xlabel="IdPhi", ylabel="Density")