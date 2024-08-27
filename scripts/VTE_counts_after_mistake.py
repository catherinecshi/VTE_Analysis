import os
import pandas as pd

from src import helper

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")
for rat in os.listdir(base_path):
    if ".DS_Store" in rat:
        continue
    
    VTEs = {}
    rat_path = os.path.join(base_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file)
            zIdPhi_csv = pd.read_csv(file_path)
            
            # get threshold above which a trial is counted as a VTE trial
            mean_zIdPhi = zIdPhi_csv["zIdPhi"].mean
            std_zIdPhi = zIdPhi_csv["zIdPhi"].std
            VTE_threshold = mean_zIdPhi + std_zIdPhi
            
            last_correct = None
            for index, row in zIdPhi_csv.iterrows():
                is_VTE = row["zIdPhi"] > VTE_threshold
                is_correct = row["Correct"]
                
                if is_VTE:
                    