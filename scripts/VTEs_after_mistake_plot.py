import os
import math
import starbars
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

from models import helper

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")

until_next_mistake = {} # {traj_id: # trials until next mistake}
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
                
                mistakes = []
                for index, row in trajectory_csv.iterrows():
                    traj_id = row["ID"]
                    parts = traj_id.split("_")
                    traj_number = parts[2]
                    is_correct = row["Correct"]
                    
                    if not is_correct:
                        mistakes.append(int(traj_number))
                
                if not mistakes:
                    continue
                else:
                    for index, row in trajectory_csv.iterrows():
                        traj_id = row["ID"]
                        parts = traj_id.split("_")
                        traj_number = parts[2]
                        larger_numbers = [num for num in mistakes if num < int(traj_number)]
                        
                        if int(traj_number) in mistakes:
                            until_next_mistake[traj_id] = 0
                        elif larger_numbers:
                            closest_number = max(larger_numbers)
                            if (closest_number - int(traj_number)) == 0:
                                print(traj_id)
                            until_next_mistake[traj_id] = int(traj_number) - closest_number


trues = [] # trials until mistake for VTE trials
falses = []
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
            
            # go through each day independently
            grouped_by_day = zIdPhi_csv.groupby("Day")
            for day, group in grouped_by_day:
                last_mistake = None
                since_last_mistake = None
                for index, row in group.iterrows():
                    traj_id = row["ID"]
                    is_VTE = row["zIdPhi"] > VTE_threshold
                    
                    try:
                        trials_until_mistake = until_next_mistake[traj_id]
                    except KeyError:
                        #print(f"{traj_id} not found in trials_until_mistake")
                        continue
                    
                    if is_VTE:
                        trues.append(trials_until_mistake)
                    else:
                        falses.append(trials_until_mistake)
            
trues_count_no_trials = {value: trues.count(value) for value in set(trues)} # how many trues for specific # trials
falses_count_no_trials = {value: falses.count(value) for value in set(falses)}

VTE_vs_last_mistake = {} # {number of trials until mistake: VTE_proportion}
n_0 = None
n_1 = None
p_0 = None
p_1 = None
above_2 = {"n": [], "p": []}
for no_trials in trues_count_no_trials.keys():
    # see if this is 2+
    if no_trials >= 2:
        if no_trials in falses_count_no_trials:
            trues_count = trues_count_no_trials[no_trials]
            falses_count = falses_count_no_trials[no_trials]
            
            if trues_count + falses_count < 10:
                continue
            
            above_2["n"].append(falses_count)
            above_2["p"].append(trues_count)
        else:
            continue
    # this is for 0 or 1
    elif no_trials in falses_count_no_trials:
        trues_count = trues_count_no_trials[no_trials]
        falses_count = falses_count_no_trials[no_trials]
        
        # make sure there is enough trials for statistical analysis
        if trues_count + falses_count < 10:
            continue
        
        VTE_proportion = (trues_count / (trues_count + falses_count)) * 100
        VTE_vs_last_mistake[no_trials] = VTE_proportion
        
        if no_trials == 0:
            n_0 = trues_count + falses_count
            p_0 = trues_count / (trues_count + falses_count)
        elif no_trials == 1:
            n_1 = trues_count + falses_count
            p_1 = trues_count / (trues_count + falses_count)
        else:
            print(f"something weird, no_trials not 0 or 1")
    else:
        print(f"something weird, all trials are VTEs for {no_trials}")
        continue

# aggregate 2+
n_sum = sum(above_2["n"])
p_sum = sum(above_2["p"])
p_sum_new = (p_sum / (p_sum + n_sum)) * 100
VTE_vs_last_mistake[int(2)] = p_sum_new

# comparing 0 and 1
z_scores = []
z_scores_new = []
p_vals = []
if p_0 and p_1 and n_0 and n_1:
    d = p_0 - p_1
    p = (p_0 * n_0 + p_1 * n_1) / n_0 + n_1
    pooled_std = p * (1 - p) * (1 / n_0 + 1 / n_1) ** 0.05
    
    # zscore
    z_score = d / pooled_std
    z_scores.append(z_score)
    
    # alternative zscore bc
    count_0 = p_0 * n_0
    count_1 = p_1 * n_1
    count = np.array([count_0, count_1])
    nobs = np.array([n_0, n_1])
    z_score_new, pval = sm.stats.proportions_ztest(count, nobs)
    z_scores_new.append(z_score)
    p_vals.append(pval)
else:
    print("something missing for 0 and 1")

if p_1 and p_sum and n_1 and n_sum:
    d = p_1 - p_sum
    p = (p_1 * n_1 + p_sum) / n_1 + (p_sum + n_sum)
    pooled_std = p * (1 - p) * (1 / n_1 + 1 / (p_sum + n_sum)) ** 0.05
    
    # zscore
    z_score = d / pooled_std
    z_scores.append(z_score)
    
    count_2 = p_sum
    count_1 = p_1 * n_1
    count = np.array([count_1, count_2])
    nobs = np.array([n_1, (n_sum + p_sum)])
    z_score_new, pval = sm.stats.proportions_ztest(count, nobs)
    z_scores_new.append(z_score)
    p_vals.append(pval)
else:
    print("something missing for 1 and sum")

if p_0 and p_sum and n_0 and n_sum:
    d = p_0 - p_sum
    p = (p_0 * n_0 + p_sum) / n_0 + (p_sum + n_sum)
    pooled_std = p * (1 - p) * (1 / n_0 + 1 / (p_sum + n_sum)) ** 0.05
    
    # zscore
    z_score = d / pooled_std
    z_scores.append(z_score)
    
    count_0 = p_0 * n_0
    count_2 = p_sum
    count = np.array([count_0, count_2])
    nobs = np.array([n_0, (p_sum + n_sum)])
    z_score_new, pval = sm.stats.proportions_ztest(count, nobs)
    z_scores_new.append(z_score)
    p_vals.append(pval)
else:
    print("something missing for 0 and sum")

print(z_scores)
print(z_scores_new)
print(p_vals)
critical_z_score = stats.norm.ppf(0.975)

significance = []
for z_score in z_scores:
    if z_score > critical_z_score:
        significance.append(True)
    else:
        significance.append(False)

def create_bar_plot(data, x_ticks, xlim=None, ylim=None, pvals=None, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    plt.bar(x_ticks, data)
    
    x_ticks = ["0", "1", "2+"]
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks([0, 1, 2], labels=x_ticks, fontsize=20)
    plt.yticks(fontsize=20)
    
    if xlim:
        plt.xlim(xlim)
    
    if ylim:
        plt.ylim(ylim)
        
    # draw significance annotations
    annotations = [(1, 2, pvals[1]), (0, 2, pvals[2])]
    starbars.draw_annotation(annotations)
    
    plt.tight_layout()
    plt.show()

x_ticks = VTE_vs_last_mistake.keys()
data = VTE_vs_last_mistake.values()
create_bar_plot(data, x_ticks, xlim=(-0.5, 2.5), pvals=p_vals,
                title="VTEs after Mistakes",
                xlabel="Number of Trials After a Mistake will Occur",
                ylabel=r"% of VTE Trials")
