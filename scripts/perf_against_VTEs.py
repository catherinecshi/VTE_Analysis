import os
import numpy as np
import pandas as pd


# Initialize lists to store data
data = []
temp_path = "/Users/catpillow/Documents/VTE_Analysis/"

# Get all rat folders
rat_folders = [f for f in os.listdir(os.path.join(temp_path, "processed_data/VTE_values")) 
               if os.path.isdir(os.path.join(temp_path, "processed_data/VTE_values", f))]

for rat in rat_folders:
    if rat == "BP07":
        continue #dunno why zidphi value has bp09's data
    
    # Read zIdPhi data for this rat
    zidphi_path = os.path.join(temp_path, "processed_data/VTE_values", rat, "zIdPhis.csv")
    zidphi_df = pd.read_csv(zidphi_path)
    
    # Calculate mean and std of zIdPhi values
    mean_zidphi = np.mean(zidphi_df["zIdPhi"])
    std_zidphi = np.std(zidphi_df["zIdPhi"])
    vte_threshold = mean_zidphi + (1.5 * std_zidphi)
    
    # Get unique days
    unique_days = zidphi_df["Day"].unique()
    
    for day in unique_days:
        # Get data for this day
        day_zidphi = zidphi_df[zidphi_df["Day"] == day]
        
        # Calculate VTE percentage for this day
        total_trials = len(day_zidphi)
        vte_trials = len(day_zidphi[day_zidphi["zIdPhi"] > vte_threshold])
        vte_percentage = (vte_trials / total_trials) * 100
        
        # Get trajectory data for this day
        traj_path = os.path.join(temp_path, "processed_data/VTE_values", rat, str(day), "trajectories.csv")
        traj_df = pd.read_csv(traj_path)
        
        # Calculate performance percentage
        correct_trials = len(traj_df[traj_df["Correct"] == True])
        total_trials = len(traj_df)
        performance_percentage = (correct_trials / total_trials) * 100
        
        # Add to data list
        data.append({
            "Rat": rat,
            "Day": day,
            "VTE_Percentage": vte_percentage,
            "Performance_Percentage": performance_percentage
        })

# Create DataFrame and save to CSV
results_df = pd.DataFrame(data)
results_df.to_csv("/Users/catpillow/Documents/VTE_Analysis/processed_data/VTE_vs_Performance.csv", index=False)
