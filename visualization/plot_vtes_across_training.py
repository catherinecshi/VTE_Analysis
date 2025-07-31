import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats

from config.paths import paths

def calculate_daily_performance(df, trial_type, day):
    """Calculate performance (% correct) for a specific trial type on a specific day"""
    day_data = df[(df['Day'] == day) & (df['Trial_Type'] == trial_type)]
    if len(day_data) == 0:
        return np.nan
    return (day_data['Correct'].sum() / len(day_data)) * 100

def find_criteria_day(df, trial_type, first_day):
    """Find the day when criteria is met (75% for 2 consecutive days)"""
    # Get all days for this trial type
    trial_days = sorted(df[df['Trial_Type'] == trial_type]['Day'].unique())
    trial_days = [day for day in trial_days if day >= first_day]
    
    if len(trial_days) < 2:
        return None
    
    for i in range(len(trial_days) - 1):
        day1, day2 = trial_days[i], trial_days[i + 1]
        
        # Calculate performance for both days
        perf1 = calculate_daily_performance(df, trial_type, day1)
        perf2 = calculate_daily_performance(df, trial_type, day2)
        
        # Check if both days meet criteria (75% and consecutive)
        if (not np.isnan(perf1) and not np.isnan(perf2) and 
            perf1 >= 75 and perf2 >= 75 and day2 == day1 + 1):
            return day1  # Return the first day of the two consecutive days
    
    return None

def get_average_zidphi(df, trial_type, day):
    """Get average zIdPhi for a specific trial type on a specific day"""
    day_data = df[(df['Day'] == day) & (df['Trial_Type'] == trial_type) & (df['zIdPhi'] != 0)]
    if len(day_data) == 0:
        return np.nan
    return day_data['zIdPhi'].mean()

def process_rat_data(csv_path):
    """Process a single rat's CSV file"""
    try:
        df = pd.read_csv(csv_path)
        rat_id = csv_path.parent.name  # Assuming folder name is rat ID
        
        # Handle Day column - extract numeric part if it contains text like "Day12"
        if df['Day'].dtype == 'object':  # If Day column contains strings
            df['Day'] = df['Day'].astype(str).str.extract('(\d+)').astype(int)
        else:
            df['Day'] = df['Day'].astype(int)
        
        # Ensure Trial_Type is integer
        df['Trial_Type'] = df['Trial_Type'].astype(int)
        
        # Ensure Correct is boolean
        if df['Correct'].dtype == 'object':
            df['Correct'] = df['Correct'].map({'TRUE': True, 'FALSE': False})
        else:
            df['Correct'] = df['Correct'].astype(bool)
        
        results = {}
        trial_types = sorted(df['Trial_Type'].unique())
        
        for trial_type in trial_types:
            # Find first day of this trial type
            first_day = df[df['Trial_Type'] == trial_type]['Day'].min()
            
            # Find criteria day
            criteria_day = find_criteria_day(df, trial_type, first_day)
            
            if criteria_day is None:
                print(f"Rat {rat_id}, Trial Type {trial_type}: Never met 75% criteria")
                # Still get first day data
                first_zidphi = get_average_zidphi(df, trial_type, first_day)
                results[trial_type] = {
                    'first_day': first_day,
                    'first_zidphi': first_zidphi,
                    'middle_day': None,
                    'middle_zidphi': np.nan,
                    'criteria_day': None,
                    'criteria_zidphi': np.nan,
                    'complete': False
                }
            else:
                # Calculate middle day
                middle_day = int((first_day + criteria_day) / 2)
                
                # Get zIdPhi values
                first_zidphi = get_average_zidphi(df, trial_type, first_day)
                middle_zidphi = get_average_zidphi(df, trial_type, middle_day)
                criteria_zidphi = get_average_zidphi(df, trial_type, criteria_day)
                
                # Check if middle day has data
                if np.isnan(middle_zidphi):
                    print(f"Rat {rat_id}, Trial Type {trial_type}: No data for middle day {middle_day}")
                    results[trial_type] = {
                        'first_day': first_day,
                        'first_zidphi': first_zidphi,
                        'middle_day': None,
                        'middle_zidphi': np.nan,
                        'criteria_day': criteria_day,
                        'criteria_zidphi': criteria_zidphi,
                        'complete': False
                    }
                else:
                    results[trial_type] = {
                        'first_day': first_day,
                        'first_zidphi': first_zidphi,
                        'middle_day': middle_day,
                        'middle_zidphi': middle_zidphi,
                        'criteria_day': criteria_day,
                        'criteria_zidphi': criteria_zidphi,
                        'complete': True
                    }
        
        return rat_id, results
    
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None, None

def create_plot(all_rat_data):
    """Create the main plot"""
    # Get all trial types
    all_trial_types = set()
    for rat_data in all_rat_data.values():
        all_trial_types.update(rat_data.keys())
    trial_types = sorted(list(all_trial_types))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set up x-axis positions
    time_points = ['first', 'middle', 'criteria']
    time_labels = ['First Day', 'Middle Day', 'Criteria Day']
    
    # Each trial type gets 3 positions (first, middle, criteria) plus some spacing
    positions_per_trial = 3
    spacing_between_trials = 1
    total_width_per_trial = positions_per_trial + spacing_between_trials
    
    all_x_positions = []
    all_x_labels = []
    trial_centers = []
    
    for i, trial_type in enumerate(trial_types):
        # Calculate x positions for this trial type
        trial_start = i * total_width_per_trial
        trial_positions = [trial_start + j for j in range(positions_per_trial)]
        all_x_positions.extend(trial_positions)
        all_x_labels.extend(time_labels)
        trial_centers.append(trial_start + 1)  # Center of the three positions
        
        # Collect data for this trial type
        trial_data = {'first': [], 'middle': [], 'criteria': []}
        individual_data = {'first': {}, 'middle': {}, 'criteria': {}}
        
        for rat_id, rat_data in all_rat_data.items():
            if trial_type in rat_data:
                for j, time_point in enumerate(time_points):
                    if time_point == 'first':
                        val = rat_data[trial_type]['first_zidphi']
                    elif time_point == 'middle':
                        val = rat_data[trial_type]['middle_zidphi']
                    else:  # criteria
                        val = rat_data[trial_type]['criteria_zidphi']
                    
                    if not np.isnan(val):
                        trial_data[time_point].append(val)
                        individual_data[time_point][rat_id] = val
        
        # Plot individual rat trajectories (thin lines within each trial type)
        for rat_id in all_rat_data.keys():
            if trial_type in all_rat_data[rat_id]:
                rat_x = []
                rat_y = []
                
                for j, time_point in enumerate(time_points):
                    if rat_id in individual_data[time_point]:
                        rat_x.append(trial_positions[j])
                        rat_y.append(individual_data[time_point][rat_id])
                
                if len(rat_x) >= 2:  # Only plot if we have at least 2 points
                    ax.plot(rat_x, rat_y, color='gray', alpha=0.3, linewidth=1, zorder=1)
        
        # Plot means with error bars for each time point
        colors = ['blue', 'orange', 'green']
        for j, time_point in enumerate(time_points):
            if trial_data[time_point]:
                mean_val = np.mean(trial_data[time_point])
                sem_val = stats.sem(trial_data[time_point])
                
                ax.errorbar(trial_positions[j], mean_val, yerr=sem_val,
                           color=colors[j], marker='o', linewidth=2,
                           markersize=8, capsize=5, zorder=3,
                           label=time_labels[j] if i == 0 else "")  # Only label once
        
        # Connect the means with a thicker line
        mean_x = []
        mean_y = []
        for j, time_point in enumerate(time_points):
            if trial_data[time_point]:
                mean_x.append(trial_positions[j])
                mean_y.append(np.mean(trial_data[time_point]))
        
        if len(mean_x) >= 2:
            ax.plot(mean_x, mean_y, color='black', linewidth=3, alpha=0.7, zorder=2)
    
    # Customize plot
    ax.set_xlabel('Trial Type and Time Point', fontsize=12)
    ax.set_ylabel('Average zIdPhi', fontsize=12)
    ax.set_title('zIdPhi Values Across Trial Types and Time Points', fontsize=14)
    
    # Set x-axis ticks and labels
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(all_x_labels, rotation=45, ha='right')
    
    # Add trial type labels at the top
    for i, trial_type in enumerate(trial_types):
        ax.text(trial_centers[i], ax.get_ylim()[1] * 1.05, f'Trial Type {trial_type}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add vertical lines to separate trial types
    for i in range(1, len(trial_types)):
        separator_x = trial_centers[i] - (positions_per_trial + spacing_between_trials) / 2
        ax.axvline(x=separator_x, color='lightgray', linestyle='--', alpha=0.5)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main(data_path):
    """Main function to run the analysis"""
    data_path = Path(data_path)
    all_rat_data = {}
    
    # Find all rat folders and process CSV files
    for rat_folder in data_path.iterdir():
        if rat_folder.is_dir():
            if "BP06" in rat_folder.name or "BP08" in rat_folder.name or "BP09" in rat_folder.name:
                continue
            
            csv_file = rat_folder / 'zIdPhis.csv'
            if csv_file.exists():
                rat_id, rat_data = process_rat_data(csv_file)
                if rat_id and rat_data:
                    all_rat_data[rat_id] = rat_data
                    print(f"Processed rat {rat_id}")
            else:
                print(f"No zIdPhis.csv found in {rat_folder}")
    
    if not all_rat_data:
        print("No data found! Check your path and file names.")
        return None
    
    print(f"\nProcessed {len(all_rat_data)} rats total")
    
    # Create plot
    fig = create_plot(all_rat_data)
    plt.show()
    
    return all_rat_data, fig

# Example usage:
# Replace 'your_data_path' with the actual path to your data
data_path = paths.vte_values
all_data, figure = main(data_path)