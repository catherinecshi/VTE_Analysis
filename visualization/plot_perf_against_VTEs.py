import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Initialize lists to store data
data = []
temp_path = "/Users/catpillow/Documents/VTE_Analysis/"

# Get all rat folders
rat_folders = [f for f in os.listdir(os.path.join(temp_path, "processed_data/VTE_values")) 
               if os.path.isdir(os.path.join(temp_path, "processed_data/VTE_values", f))]

for rat in rat_folders:
    if rat == "inferenceTesting" or rat == "convex" or rat == ".DS_Store" or rat == "BP06" or rat == "BP08" or rat == "BP09":
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
        traj_path = os.path.join(temp_path, "processed_data/VTE_values", rat, str(day), f"{rat}_{str(day)}_trajectories.csv")
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

# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# PLOT 1: Log VTE vs Performance (original plot)
# Filter out zero values for log scale
nonzero_data = results_df[results_df['VTE_Percentage'] > 0]
log_vte = np.log10(nonzero_data['VTE_Percentage'])

# Plot data points for each rat
for rat in results_df['Rat'].unique():
    rat_data = nonzero_data[nonzero_data['Rat'] == rat]
    ax1.scatter(np.log10(rat_data['VTE_Percentage']), 
              rat_data['Performance_Percentage'],
              label=rat,
              alpha=0.6)

# Add trendline for log scale
z_log = np.polyfit(log_vte, nonzero_data['Performance_Percentage'], 1)
p_log = np.poly1d(z_log)

# Calculate R-squared for log scale
correlation_matrix_log = np.corrcoef(log_vte, nonzero_data['Performance_Percentage'])
r_squared_log = correlation_matrix_log[0,1]**2

# Calculate stats for log plot
r_value_log, p_value_log = stats.pearsonr(log_vte, nonzero_data['Performance_Percentage'])
n_log = len(log_vte)

print(f"Log VTE vs Performance Stats:")
print(f"Correlation coefficient (r): {r_value_log:.3f}")
print(f"R-squared (R²): {r_squared_log:.3f}")
print(f"p-value: {p_value_log:.5f}")
print(f"Sample size (n): {n_log}")
print(f"Regression equation: y = {z_log[0]:.2f}x + {z_log[1]:.2f}")

# Add trendline to log plot
x_trendline_log = np.array([log_vte.min(), log_vte.max()])
ax1.plot(x_trendline_log, p_log(x_trendline_log), "r--", 
       label=f'y = {z_log[0]:.2f}x + {z_log[1]:.2f}\nR² = {r_squared_log:.3f}')

# Customize log plot
ax1.set_ylabel('Performance (%)', fontsize=20)
ax1.set_xlabel('Log(VTE Percentage)', fontsize=20)
ax1.set_title('Performance vs Log(VTE Percentage)', fontsize=24)
ax1.tick_params(axis='both', labelsize=16)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# PLOT 2: Normal VTE vs Performance (new plot)
# Plot data points for each rat using normal VTE percentages
for rat in results_df['Rat'].unique():
    rat_data = results_df[results_df['Rat'] == rat]
    ax2.scatter(rat_data['VTE_Percentage'], 
              rat_data['Performance_Percentage'],
              label=rat,
              alpha=0.6)

# Add trendline for normal VTE
z_normal = np.polyfit(results_df['VTE_Percentage'], results_df['Performance_Percentage'], 1)
p_normal = np.poly1d(z_normal)

# Calculate R-squared for normal VTE
correlation_matrix_normal = np.corrcoef(results_df['VTE_Percentage'], results_df['Performance_Percentage'])
r_squared_normal = correlation_matrix_normal[0,1]**2

# Calculate stats for normal plot
r_value_normal, p_value_normal = stats.pearsonr(results_df['VTE_Percentage'], results_df['Performance_Percentage'])
n_normal = len(results_df)

print(f"\nNormal VTE vs Performance Stats:")
print(f"Correlation coefficient (r): {r_value_normal:.3f}")
print(f"R-squared (R²): {r_squared_normal:.3f}")
print(f"p-value: {p_value_normal:.5f}")
print(f"Sample size (n): {n_normal}")
print(f"Regression equation: y = {z_normal[0]:.2f}x + {z_normal[1]:.2f}")

# Add trendline to normal plot
x_trendline_normal = np.array([results_df['VTE_Percentage'].min(), results_df['VTE_Percentage'].max()])
ax2.plot(x_trendline_normal, p_normal(x_trendline_normal), "r--", 
       label=f'y = {z_normal[0]:.2f}x + {z_normal[1]:.2f}\nR² = {r_squared_normal:.3f}')

# Customize normal plot
ax2.set_ylabel('Performance (%)', fontsize=20)
ax2.set_xlabel('VTE Percentage', fontsize=20)
ax2.set_title('Performance vs VTE Percentage', fontsize=24)
ax2.tick_params(axis='both', labelsize=16)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()