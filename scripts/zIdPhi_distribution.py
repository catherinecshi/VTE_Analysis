import os
import pandas as pd

from src import helper
from src import plotting

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")

zIdPhis = []
choice_zIdPhis = {}
trial_type_zIdPhis = {}
for rat in os.listdir(base_path):
    rat_path = os.path.join(base_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file)
            zIdPhi_csv = pd.read_csv(file_path)
            
            for index, row in zIdPhi_csv.iterrows():
                zIdPhi = row["zIdPhi"]
                choice = row["Choice"]
                trial_type = row["Trial_Type"]
                
                zIdPhis.append(zIdPhi)
                
                # check if key exists before adding to dictionary
                if choice in choice_zIdPhis.keys():
                    choice_zIdPhis[choice].append(zIdPhi)
                else:
                    choice_zIdPhis[choice] = []
                    choice_zIdPhis[choice].append(zIdPhi)
                
                if trial_type not in trial_type_zIdPhis.keys():
                    trial_type_zIdPhis[trial_type] = []
                trial_type_zIdPhis[trial_type].append(zIdPhi)

# make figure for counts of durations
plotting.create_frequency_histogram(zIdPhis, title="zIdPhi Distribution", stat="count",
                                    binwidth=0.1, xlabel="zIdPhi", ylabel="Counts", xlim=(-2.5, 8))
plotting.create_multiple_frequency_histograms(choice_zIdPhis, title="zIdPhi Distribution by Choice Arm",
                                              binwidth=0.1, xlabel="zIdPhi", ylabel="Density", xlim=(-2.5, 8))
plotting.create_multiple_frequency_histograms(trial_type_zIdPhis, title="zIdPhi Distribution by Trial Type",
                                              binwidth=0.1, xlabel="zIdPhi", ylabel="Density", xlim=(-2.5, 8))