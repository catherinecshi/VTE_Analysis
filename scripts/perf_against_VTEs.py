import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Initialize lists to store data
data = []
temp_path = "/Users/catpillow/Documents/VTE_Analysis/"

# Get all rat folders
rat_folders = [f for f in os.listdir(os.path.join(temp_path, "processed_data/VTE_values")) 
               if os.path.isdir(os.path.join(temp_path, "processed_data/VTE_values", f))]

for rat in rat_folders:
    if rat == "BP07" or rat == "convex" or rat == ".DS_Store":
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

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# First subplot - Regular scale
for rat in results_df['Rat'].unique():
    rat_data = results_df[results_df['Rat'] == rat]
    ax1.scatter(rat_data['VTE_Percentage'], 
               rat_data['Performance_Percentage'],
               label=rat,
               alpha=0.6)

# Add trendline for regular scale
z1 = np.polyfit(results_df['VTE_Percentage'], results_df['Performance_Percentage'], 1)
p1 = np.poly1d(z1)

# Calculate R-squared for regular scale
correlation_matrix1 = np.corrcoef(results_df['VTE_Percentage'], results_df['Performance_Percentage'])
r_squared1 = correlation_matrix1[0,1]**2

# Add trendline to first plot
x_trendline1 = np.array([results_df['VTE_Percentage'].min(), results_df['VTE_Percentage'].max()])
ax1.plot(x_trendline1, p1(x_trendline1), "r--", 
         label=f'y = {z1[0]:.2f}x + {z1[1]:.2f}\nR² = {r_squared1:.3f}')

# Customize first plot
ax1.set_ylabel('Performance (%)')
ax1.set_xlabel('VTE Percentage (%)')
ax1.set_title('Performance vs VTE Percentage')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# ... existing code ...

# Second subplot - Log scale
# Filter out zero values
nonzero_data = results_df[results_df['VTE_Percentage'] > 0]
log_vte = np.log10(nonzero_data['VTE_Percentage'])
for rat in results_df['Rat'].unique():
    rat_data = nonzero_data[nonzero_data['Rat'] == rat]
    ax2.scatter(np.log10(rat_data['VTE_Percentage']), 
               rat_data['Performance_Percentage'],
               label=rat,
               alpha=0.6)

# Add trendline for log scale
z2 = np.polyfit(log_vte, nonzero_data['Performance_Percentage'], 1)
p2 = np.poly1d(z2)

# Calculate R-squared for log scale
correlation_matrix2 = np.corrcoef(log_vte, nonzero_data['Performance_Percentage'])
r_squared2 = correlation_matrix2[0,1]**2

# Add trendline to second plot
x_trendline2 = np.array([log_vte.min(), log_vte.max()])
ax2.plot(x_trendline2, p2(x_trendline2), "r--", 
         label=f'y = {z2[0]:.2f}x + {z2[1]:.2f}\nR² = {r_squared2:.3f}')

# ... existing code ...

# Customize second plot
ax2.set_ylabel('Performance (%)')
ax2.set_xlabel('Log(VTE Percentage)')
ax2.set_title('Performance vs Log(VTE Percentage)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
