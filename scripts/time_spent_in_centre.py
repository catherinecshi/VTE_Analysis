import os
import numpy as np
import pandas as pd

from src import helper
from src import plotting

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")

traj_lens = []
correct_traj_lens = []
incorrect_traj_lens = []
ids_of_lengths = {}
for rat in os.listdir(base_path):
    if ".DS_Store" in rat:
        continue
    
    rat_path = os.path.join(base_path, rat)
    for day in os.listdir(rat_path):
        if ".DS_Store" in day:
            continue
        
        day_path = os.path.join(rat_path, day)
        for root, _, files in os.walk(day_path):
            for file in files:
                if "trajectories" not in file:
                    continue

                file_path = os.path.join(root, file)
                trajectory_csv = pd.read_csv(file_path)
                
                for index, row in trajectory_csv.iterrows():
                    traj_id = row["ID"]
                    is_correct = row["Correct"]
                    traj_len = row["Length"]
                    traj_lens.append(traj_len)
                    
                    if is_correct:
                        correct_traj_lens.append(traj_len)
                    else:
                        incorrect_traj_lens.append(traj_len)
                    
                    # store so it can be matched up with VTE data
                    ids_of_lengths[traj_id] = traj_len

VTE_lengths = []
non_VTE_lengths = []
for rat in os.listdir(base_path):
    rat_path = os.path.join(base_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file)
            zIdPhi_csv = pd.read_csv(file_path)
            
            mean_zIdPhi = np.mean(zIdPhi_csv["zIdPhi"])
            std_zIdPhi = np.std(zIdPhi_csv["zIdPhi"])
            VTE_threshold = mean_zIdPhi + (std_zIdPhi * 1.5)
            
            for index, row in zIdPhi_csv.iterrows():
                traj_id = row["ID"]
                is_VTE = row["zIdPhi"] > VTE_threshold
                
                try:
                    traj_len = ids_of_lengths[traj_id]
                except KeyError:
                    print(f"no length for {traj_id}")
                    continue

                if is_VTE:
                    VTE_lengths.append(traj_len)
                else:
                    non_VTE_lengths.append(traj_len)

import scipy.stats as stats

# Print the sample sizes
print("\n----- T-Test Analysis: VTE vs Non-VTE Trajectory Lengths -----")
print(f"Number of VTE trajectories: {len(VTE_lengths)}")
print(f"Number of non-VTE trajectories: {len(non_VTE_lengths)}")

# Calculate basic statistics
vte_mean = np.mean(VTE_lengths)
vte_std = np.std(VTE_lengths)
vte_median = np.median(VTE_lengths)
vte_sem = vte_std / np.sqrt(len(VTE_lengths))  # Standard error of the mean

non_vte_mean = np.mean(non_VTE_lengths)
non_vte_std = np.std(non_VTE_lengths)
non_vte_median = np.median(non_VTE_lengths)
non_vte_sem = non_vte_std / np.sqrt(len(non_VTE_lengths))

# Print descriptive statistics
print("\nDescriptive Statistics:")
print(f"VTE trajectories:")
print(f"  - Mean: {vte_mean:.4f} seconds")
print(f"  - Median: {vte_median:.4f} seconds")
print(f"  - Standard deviation: {vte_std:.4f} seconds")
print(f"  - Standard error: {vte_sem:.4f} seconds")

print(f"\nNon-VTE trajectories:")
print(f"  - Mean: {non_vte_mean:.4f} seconds")
print(f"  - Median: {non_vte_median:.4f} seconds")
print(f"  - Standard deviation: {non_vte_std:.4f} seconds") 
print(f"  - Standard error: {non_vte_sem:.4f} seconds")

print(f"\nMean difference (VTE - non-VTE): {vte_mean - non_vte_mean:.4f} seconds")

# Perform the t-test (Welch's t-test which doesn't assume equal variances)
t_stat, p_value = stats.ttest_ind(VTE_lengths, non_VTE_lengths, equal_var=False)
print(f"\nWelch's t-test: t = {t_stat:.4f}, p = {p_value:.6f}")

# make figure for counts of durations
plotting.create_frequency_histogram(traj_lens, xlim=(0,6), title="Time Spent in Centre Zone",
                                    binwidth=0.1, xlabel="Time (s)", ylabel="Density")
plotting.create_frequency_histogram(correct_traj_lens, label1="Correct Trials", xlim=(0,6),
                                    list2=incorrect_traj_lens, label2="Incorrect Trials",
                                    title="Time Spent in Centre Zone during Correct vs Incorrect Trials",
                                    binwidth=0.1, xlabel="Time (s)", ylabel="Density")
plotting.create_frequency_histogram(VTE_lengths, label1="VTE Trials", xlim=(0,5),
                                    list2=non_VTE_lengths, label2="Non-VTE Trials",
                                    title="Time Spent in Centre Zone during VTE vs Non-VTE Trials",
                                    binwidth=0.1, xlabel="Time (s)", ylabel="Density")

# make cumulative frequency figure
plotting.plot_cumulative_frequency(traj_lens, title="Time Spent in Centre Zone",
                                   xlabel="Time (s)", ylabel="Cumulative Frequency")