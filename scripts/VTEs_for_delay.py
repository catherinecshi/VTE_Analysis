import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import helper
from src import plotting
from src import data_processing
from src import performance_analysis

data_path = os.path.join(helper.BASE_PATH, "data", "VTE_Data")
data_structure = data_processing.load_data_structure(data_path)

days_since_new_arm = performance_analysis.get_days_since_new_arm(data_path, data_structure)
days_since_new_arm["trials_available"] = days_since_new_arm["trials_available"].apply(lambda x: [int (y) for y in x])
days_since_new_arm = days_since_new_arm.astype({"rat": "str",
                                                "day": "int", 
                                                "arm_added": "bool", 
                                                "days_since_new_arm": "int"})

vtes_during_volatility = []

vte_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values")
for rat in os.listdir(vte_path):
    if ".DS" in rat:
        continue

    rat_path = os.path.join(vte_path, rat)
    for root, _, files in os.walk(rat_path):
        for f in files:
            if "zIdPhi" not in f:
                continue
            
            file_path = os.path.join(root, f)
            zIdPhi_df = pd.read_csv(file_path)
            
            grouped_by_day = zIdPhi_df.groupby(by="Day")
            for day, day_group in grouped_by_day:
                vte_trials = day_group[day_group["zIdPhi"] >= 1]
                non_vte_trials = day_group[day_group["zIdPhi"] < 1]
                no_vtes = len(vte_trials)
                no_non_vtes = len(non_vte_trials)
                perc_vtes = (no_vtes / (no_vtes + no_non_vtes)) * 100
                
                match = re.search(r"\d+", day)
                if match:
                    day_number = int(match.group())
                    
                corresponding_row = days_since_new_arm[(days_since_new_arm["rat"] == rat) &
                                                       (days_since_new_arm["day"] == day_number)]
                
                if not corresponding_row.empty:
                    no_days = corresponding_row["days_since_new_arm"].values[0]
                
                vtes_during_volatility.append({"rat": rat, "day": day,
                                              "perc_vtes": perc_vtes, "no_days": no_days})
                
vtes_during_volatility_df = pd.DataFrame(vtes_during_volatility)
vtes_during_volatility_df.to_csv(os.path.join(vte_path, "vtes_during_volatility.csv"))
print(vtes_during_volatility_df["no_days"].dtype, "no days")
print(vtes_during_volatility_df["perc_vtes"].dtype, "perc vtes")

# Calculate the mean and SEM for each day
mean_perc_vtes = vtes_during_volatility_df.groupby("no_days")["perc_vtes"].mean()
sem_perc_vtes = vtes_during_volatility_df.groupby("no_days")["perc_vtes"].sem()

# Create the line plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(mean_perc_vtes.index, mean_perc_vtes, yerr=sem_perc_vtes, fmt='-o', capsize=5, label='% VTE Trials')
plt.title("VTEs during Volatility", fontsize=20)
plt.xlabel("Number of Days Since New Arm Added", fontsize=16)
plt.ylabel("% VTE Trials", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Define fictional data
x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mean_perc_vtes = np.array([3.5, 5, 4.8, 4.6, 4.3, 3.9, 3.5, 3.3, 2.8, 2])
sem_perc_vtes = np.array([0.5, 0.8, 0.7, 0.4, 0.36, 0.3, 0.28, 0.26, 0.25, 0.22])

# Create the line plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(x_values, mean_perc_vtes, yerr=sem_perc_vtes, fmt='-o', capsize=5, label='% VTE Trials')
plt.title("Expected VTEs during Volatility", fontsize=20)
plt.xlabel("Number of Days Since New Arm Added", fontsize=16)
plt.ylabel("% VTE Trials", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

"""
plotting.create_box_and_whisker_plot(vtes_during_volatility_df, x="no_days", y="perc_vtes",
                                    title="VTEs during Volatility",
                                    xlabel="Number of Days Since New Arm Added",
                                    ylabel="% VTE Trials")

plotting.create_histogram(vtes_during_volatility_df, "no_days", "perc_vtes",
                          title="VTEs during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="% VTE Trials")

plotting.create_line_plot(vtes_during_volatility_df["days_since_new_arm"],
                          vtes_during_volatility_df["perc_vtes"],
                          title="VTEs during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="% VTE Trials")
                          
"""