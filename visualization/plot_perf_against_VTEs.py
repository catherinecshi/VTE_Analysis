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
    if rat == "inferenceTesting" or rat == "convex" or rat == ".DS_Store" or rat == "BP06" or rat == "BP08":
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

# Create a single figure instead of two subplots
fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted figure size for a single plot

# Filter out zero values for log scale
nonzero_data = results_df[results_df['VTE_Percentage'] > 0]
log_vte = np.log10(nonzero_data['VTE_Percentage'])

# Plot data points for each rat
for rat in results_df['Rat'].unique():
    rat_data = nonzero_data[nonzero_data['Rat'] == rat]
    ax.scatter(np.log10(rat_data['VTE_Percentage']), 
              rat_data['Performance_Percentage'],
              label=rat,
              alpha=0.6)

# Add trendline for log scale
z2 = np.polyfit(log_vte, nonzero_data['Performance_Percentage'], 1)
p2 = np.poly1d(z2)

# Calculate R-squared for log scale
correlation_matrix2 = np.corrcoef(log_vte, nonzero_data['Performance_Percentage'])
r_squared2 = correlation_matrix2[0,1]**2

# Calculate stats
r_value2, p_value2 = stats.pearsonr(log_vte, nonzero_data['Performance_Percentage'])
n = len(log_vte)  # Sample size

print(f"Log VTE vs Performance Stats:")
print(f"Correlation coefficient (r): {r_value2:.3f}")
print(f"R-squared (R²): {r_squared2:.3f}")
print(f"p-value: {p_value2:.5f}")
print(f"Sample size (n): {n}")
print(f"Regression equation: y = {z2[0]:.2f}x + {z2[1]:.2f}")

# Add trendline to plot
x_trendline = np.array([log_vte.min(), log_vte.max()])
ax.plot(x_trendline, p2(x_trendline), "r--", 
       label=f'y = {z2[0]:.2f}x + {z2[1]:.2f}\nR² = {r_squared2:.3f}')

# Customize plot with correct methods for tick labels
ax.set_ylabel('Performance (%)', fontsize=24)
ax.set_xlabel('Log(VTE Percentage)', fontsize=24)
ax.set_title('Performance vs Log(VTE Percentage)', fontsize=30)

# Fix the tick label font size - this is the corrected code
ax.tick_params(axis='both', labelsize=20)  # Sets both x and y tick label sizes

# Add legend with proper positioning
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()