import os
import numpy as np
import pandas as pd

base_path = "/Users/catpillow/Documents/VTE_Analysis/"
rat_folders = [f for f in os.listdir(os.path.join(base_path, "processed_data/VTE_values")) 
               if os.path.isdir(os.path.join(base_path, "processed_data/VTE_values", f))]

# get length and correctness and put in a dataframe
AB_dict = []
BC_dict = []
CD_dict = []
DE_dict = []
all_dict = []

subj_idx = 0
for rat in rat_folders:
    if rat == "BP07":
        continue
    
    # Read zIdPhi data for this rat
    zidphi_path = os.path.join(base_path, "processed_data/VTE_values", rat, "zIdPhis.csv")
    zidphi_df = pd.read_csv(zidphi_path)
    
    # Calculate mean and std of zIdPhi values
    mean_zidphi = np.mean(zidphi_df["zIdPhi"])
    std_zidphi = np.std(zidphi_df["zIdPhi"])
    vte_threshold = mean_zidphi + (1.5 * std_zidphi)
    
    # Get unique days and sort them
    unique_days = sorted(zidphi_df["Day"].unique(), key=lambda x: int(str(x).replace("Day", "")))
    
    rat_data = []
    for day in unique_days:
        # Get data for this day
        day_zidphi = zidphi_df[zidphi_df["Day"] == day]
        
        # Get trajectory data for this day
        traj_path = os.path.join(base_path, "processed_data/VTE_values", rat, str(day), "trajectories.csv")
        traj_df = pd.read_csv(traj_path)
        
        for index, row in traj_df.iterrows():
            if row["Correct"] == True:
                correct = 1
            else:
                correct = 0
                
            # get VTE or not
            traj_id = row["ID"]
            trial_zidphi = day_zidphi[day_zidphi["ID"] == traj_id]
            
            if trial_zidphi["zIdPhi"].iloc[0] > vte_threshold:
                vte = 1
            else:
                vte = 0
                
            new_day = int(str(day).replace("Day", "")) - 1
            
            rat_data.append({
                "subj_idx": new_day,
                "stim": row["Trial Type"],
                "rt": row["Length"],
                "response": correct,
                "vte": vte
            })
            
            # this is for the trial type based reaction time csv
            trial_type = row["Trial Type"]
            
            if trial_type == 1:
                AB_dict.append({
                    "subj_idx": subj_idx,
                    "rt": row["Length"],
                    "response": correct,
                    "stim": vte
                })
            elif trial_type == 2:
                BC_dict.append({
                    "subj_idx": subj_idx,
                    "rt": row["Length"],
                    "response": correct,
                    "stim": vte
                })
            elif trial_type == 3:
                CD_dict.append({
                    "subj_idx": subj_idx,
                    "rt": row["Length"],
                    "response": correct,
                    "stim": vte
                })
            elif trial_type == 4:
                DE_dict.append({
                    "subj_idx": subj_idx,
                    "rt": row["Length"],
                    "response": correct,
                    "stim": vte
                })
            
            all_dict.append({
                "subj_idx": subj_idx,
                "rt": row["Length"],
                "response": correct,
                "stim": vte,
                "trial_type": trial_type
            })
        
    results_df = pd.DataFrame(rat_data)
    rat_path = os.path.join(base_path, "processed_data", "response_time", f"{rat}_RT.csv")
    results_df.to_csv(rat_path, index=False)
    
    subj_idx += 1

AB_df = pd.DataFrame(AB_dict)
AB_path = os.path.join(base_path, "processed_data", "response_time", "AB_RT.csv")
AB_df.to_csv(AB_path, index=False)

BC_df = pd.DataFrame(BC_dict)
BC_path = os.path.join(base_path, "processed_data", "response_time", "BC_RT.csv")
BC_df.to_csv(BC_path, index=False)

CD_df = pd.DataFrame(CD_dict)
CD_path = os.path.join(base_path, "processed_data", "response_time", "CD_RT.csv")
CD_df.to_csv(CD_path, index=False)

DE_df = pd.DataFrame(DE_dict)
DE_path = os.path.join(base_path, "processed_data", "response_time", "DE_RT.csv")
DE_df.to_csv(DE_path, index=False)

all_df = pd.DataFrame(all_dict)
all_path = os.path.join(base_path, "processed_data", "response_time", "all_RT.csv")
all_df.to_csv(all_path, index=False)