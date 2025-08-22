"""
VTE Data Discrepancy Analysis Script

This script compares data between two sources:
1. VTE_values: /processed_data/VTE_values/{rat}/{day}/zIdPhi_day_{day}.csv
2. data_for_model: /processed_data/data_for_model/{rat}/Day{day_num}.csv

It identifies discrepancies in trial counts, missing trials, and VTE classification differences.
Additionally, it plots VTE distribution by trial type from the data_for_model data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re
from pathlib import Path

from config.settings import TRIAL_TYPE_MAPPINGS, TRIAL_TYPE_MAPPINGS_EF, HIERARCHY_MAPPINGS, RATS_WITH_EF

# Configuration
EXCLUDE_FOLDERS = ["Example", "inferenceTesting", ".DS_Store"]
EXCLUDE_RATS = []  # Add specific rats to exclude here, e.g., ["BP06", "BP09"]

# Filtering logic to match plot_vtes_trial_type.py
def matches_plot_vtes_filtering(rat_name):
    """Check if rat should be excluded using plot_vtes_trial_type.py logic"""
    return ".DS_Store" in rat_name or "BP0" in rat_name or "BP10" in rat_name or "inferenceTesting" in rat_name

def collect_plot_vtes_style_data(vte_path):
    """Collect VTE data using the exact same method as plot_vtes_trial_type.py"""
    VTE_trials = {}
    all_trials = {}
    trial_type_vte_counts_plot_style = defaultdict(lambda: {'vte_true': 0, 'vte_false': 0, 'total': 0})
    
    for rat in os.listdir(vte_path):
        if matches_plot_vtes_filtering(rat):
            continue
        
        rat_path = os.path.join(vte_path, rat)
        if not os.path.isdir(rat_path):
            continue
            
        for root, _, files in os.walk(rat_path):
            for file in files:
                if "zIdPhi" not in file:
                    continue

                file_path = os.path.join(root, file)
                try:
                    zIdPhi_csv = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                
                # Calculate VTE threshold exactly like plot_vtes_trial_type.py
                zIdPhi_mean = zIdPhi_csv["zIdPhi"].mean()
                zIdPhi_std = zIdPhi_csv["zIdPhi"].std()
                VTE_threshold = zIdPhi_mean + (zIdPhi_std * 1.5)
                
                for index, row in zIdPhi_csv.iterrows():
                    zIdPhi = row["zIdPhi"]
                    trial_type = row["Trial_Type"]
                    is_VTE = zIdPhi > VTE_threshold
                    
                    # Skip trial type 5 like plot_vtes_trial_type.py
                    if trial_type == 5:
                        continue
                    
                    if is_VTE:
                        if trial_type in VTE_trials:
                            VTE_trials[trial_type] += 1
                        else:
                            VTE_trials[trial_type] = 1
                    
                    if trial_type in all_trials:
                        all_trials[trial_type] += 1
                    else:
                        all_trials[trial_type] = 1
                    
                    # Convert trial type numbers to normalized format for comparison
                    # Map trial types 1-4 to AB, BC, CD, DE
                    trial_type_labels = {1: "0-1", 2: "1-2", 3: "2-3", 4: "3-4"}  # AB, BC, CD, DE
                    
                    if trial_type in trial_type_labels:
                        label = trial_type_labels[trial_type]
                        trial_type_vte_counts_plot_style[label]['total'] += 1
                        if is_VTE:
                            trial_type_vte_counts_plot_style[label]['vte_true'] += 1
                        else:
                            trial_type_vte_counts_plot_style[label]['vte_false'] += 1
    
    return {
        'VTE_trials': VTE_trials,
        'all_trials': all_trials,
        'trial_type_counts': trial_type_vte_counts_plot_style
    }

def should_exclude_folder(folder_name):
    """Check if a folder should be excluded from analysis"""
    for exclude_pattern in EXCLUDE_FOLDERS:
        if exclude_pattern in folder_name:
            return True
    return folder_name in EXCLUDE_RATS

def extract_trial_id_from_vte(full_id):
    """Extract trial ID from VTE ID format: {rat}_{day}_{ID} -> ID"""
    parts = full_id.split('_')
    if len(parts) >= 3:
        return int(parts[-1])
    return None

def load_vte_data(vte_path, rat, day):
    """Load VTE data for a specific rat and day"""
    vte_file = os.path.join(vte_path, rat, day, f"zIdPhi_day_{day}.csv")
    if not os.path.exists(vte_file):
        return None
    
    try:
        df = pd.read_csv(vte_file)
        # Extract trial IDs and create mapping
        trial_data = {}
        for _, row in df.iterrows():
            trial_id = extract_trial_id_from_vte(row['ID'])
            if trial_id is not None:
                trial_data[trial_id] = {
                    'vte': row['VTE'],
                    'trial_type': row.get('Trial_Type', None)
                }
        return trial_data
    except Exception as e:
        print(f"Error loading VTE data for {rat}/{day}: {e}")
        return None

def load_model_data(model_path, rat, day_num):
    """Load model data for a specific rat and day"""
    model_file = os.path.join(model_path, rat, f"Day{day_num}.csv")
    if not os.path.exists(model_file):
        return None
    
    try:
        df = pd.read_csv(model_file)
        # Create mapping from ID to data
        trial_data = {}
        for _, row in df.iterrows():
            trial_id = int(row['ID'])
            trial_data[trial_id] = {
                'vte': row['VTE'],
                'day': row['Day'],
                'first': row.get('first', None),
                'second': row.get('second', None),
                'correct': row.get('correct', None)
            }
        return trial_data
    except Exception as e:
        print(f"Error loading model data for {rat}/Day{day_num}: {e}")
        return None

def extract_day_number(day_string):
    """Extract numeric day from day string (e.g., 'Day12' -> 12)"""
    match = re.search(r'\d+', day_string)
    return int(match.group()) if match else None

def compare_datasets():
    """Main function to compare VTE and model datasets"""
    # Set up paths
    base_path = Path("/Users/catpillow/Documents/VTE_Analysis")
    vte_path = base_path / "processed_data" / "VTE_values"
    model_path = base_path / "processed_data" / "data_for_model"
    
    # Track statistics
    total_comparisons = 0
    discrepancies_found = 0
    vte_classification_mismatches = 0
    missing_in_vte = 0
    missing_in_model = 0
    
    # Store data for plotting
    all_model_data = []
    all_vte_data = []
    trial_type_vte_counts = defaultdict(lambda: {'vte_true': 0, 'vte_false': 0, 'total': 0})
    trial_type_vte_counts_zidphi = defaultdict(lambda: {'vte_true': 0, 'vte_false': 0, 'total': 0})
    
    print("VTE Data Discrepancy Analysis")
    print("=" * 50)
    print()
    
    # Get all rats from VTE directory
    vte_rats = [r for r in os.listdir(vte_path) if not should_exclude_folder(r) and os.path.isdir(vte_path / r)]
    model_rats = [r for r in os.listdir(model_path) if not should_exclude_folder(r) and os.path.isdir(model_path / r)]
    
    # Also collect data using plot_vtes_trial_type.py method for comparison
    plot_vtes_style_data = collect_plot_vtes_style_data(vte_path)
    
    all_rats = set(vte_rats + model_rats)
    
    print(f"Found {len(vte_rats)} rats in VTE data")
    print(f"Found {len(model_rats)} rats in model data")
    print(f"Total unique rats: {len(all_rats)}")
    print(f"Excluded folders: {EXCLUDE_FOLDERS}")
    if EXCLUDE_RATS:
        print(f"Excluded rats: {EXCLUDE_RATS}")
    print()
    
    # Compare data for each rat
    for rat in sorted(all_rats):
        print(f"Analyzing rat: {rat}")
        
        # Check if rat exists in both datasets
        rat_in_vte = rat in vte_rats
        rat_in_model = rat in model_rats
        
        if not rat_in_vte:
            print(f"  ⚠️  Rat {rat} missing from VTE data")
            continue
        if not rat_in_model:
            print(f"  ⚠️  Rat {rat} missing from model data")
            continue
        
        # Get days for this rat
        vte_rat_path = vte_path / rat
        model_rat_path = model_path / rat
        
        vte_days = [d for d in os.listdir(vte_rat_path) if not should_exclude_folder(d) and os.path.isdir(vte_rat_path / d)]
        model_files = [f for f in os.listdir(model_rat_path) if f.endswith('.csv') and f.startswith('Day') and not should_exclude_folder(f)]
        model_days = [f.replace('.csv', '').replace('Day', '') for f in model_files]
        
        # Convert day formats for comparison
        vte_day_nums = []
        for day in vte_days:
            day_num = extract_day_number(day)
            if day_num is not None:
                vte_day_nums.append((day, day_num))
        
        model_day_nums = [(f"Day{d}", int(d)) for d in model_days if d.isdigit()]
        
        # Find overlapping days
        vte_nums = {num for _, num in vte_day_nums}
        model_nums = {num for _, num in model_day_nums}
        common_days = vte_nums & model_nums
        
        if not common_days:
            print(f"  ⚠️  No overlapping days found")
            continue
        
        # Compare each day
        rat_discrepancies = 0
        for day_num in sorted(common_days):
            # Find corresponding day strings
            vte_day = next(day for day, num in vte_day_nums if num == day_num)
            
            # Load data
            vte_data = load_vte_data(vte_path, rat, vte_day)
            model_data = load_model_data(model_path, rat, day_num)
            
            if vte_data is None or model_data is None:
                continue
            
            total_comparisons += 1
            
            # Store VTE data for plotting
            for trial_id, data in vte_data.items():
                trial_info = {
                    'rat': rat,
                    'day': day_num,
                    'trial_id': trial_id,
                    'vte': data['vte'],
                    'trial_type': data.get('trial_type')
                }
                all_vte_data.append(trial_info)
                
                # Count by trial type for zIdPhi data
                if data.get('trial_type') is not None:
                    # Convert trial type number to canonical pair format using config mappings
                    trial_type_num = data['trial_type']
                    
                    # Determine which mapping to use based on rat
                    if rat in RATS_WITH_EF:
                        # Create reverse mapping from number to letter pair
                        reverse_mapping = {v: k for k, v in TRIAL_TYPE_MAPPINGS_EF.items()}
                    else:
                        reverse_mapping = {v: k for k, v in TRIAL_TYPE_MAPPINGS.items()}
                    
                    if trial_type_num in reverse_mapping:
                        letter_pair = reverse_mapping[trial_type_num]  # e.g., "AB"
                        # Convert letters to numbers using hierarchy mapping
                        first_num = HIERARCHY_MAPPINGS[letter_pair[0]]
                        second_num = HIERARCHY_MAPPINGS[letter_pair[1]]
                        
                        # Normalize by sorting the pair
                        normalized_pair = tuple(sorted([first_num, second_num]))
                        trial_type_label = f"{normalized_pair[0]}-{normalized_pair[1]}"
                        
                        trial_type_vte_counts_zidphi[trial_type_label]['total'] += 1
                        if data['vte']:
                            trial_type_vte_counts_zidphi[trial_type_label]['vte_true'] += 1
                        else:
                            trial_type_vte_counts_zidphi[trial_type_label]['vte_false'] += 1
            
            # Compare trial counts
            vte_trials = set(vte_data.keys())
            model_trials = set(model_data.keys())
            
            # Store model data for plotting
            for trial_id, data in model_data.items():
                trial_info = {
                    'rat': rat,
                    'day': day_num,
                    'trial_id': trial_id,
                    'vte': data['vte'],
                    'first': data.get('first'),
                    'second': data.get('second'),
                    'correct': data.get('correct')
                }
                all_model_data.append(trial_info)
                
                # Determine trial type for plotting (normalize to canonical form)
                first, second = data.get('first'), data.get('second')
                if first is not None and second is not None:
                    # Normalize trial type by sorting the pair (e.g., 1-0 becomes 0-1)
                    normalized_pair = tuple(sorted([first, second]))
                    trial_type = f"{normalized_pair[0]}-{normalized_pair[1]}"
                    trial_type_vte_counts[trial_type]['total'] += 1
                    if data['vte']:
                        trial_type_vte_counts[trial_type]['vte_true'] += 1
                    else:
                        trial_type_vte_counts[trial_type]['vte_false'] += 1
            
            # Check for discrepancies
            missing_in_model = vte_trials - model_trials
            missing_in_vte_data = model_trials - vte_trials
            common_trials = vte_trials & model_trials
            
            day_has_discrepancy = False
            
            if len(vte_trials) != len(model_trials):
                print(f"    Day {day_num}: Trial count mismatch - VTE: {len(vte_trials)}, Model: {len(model_trials)}")
                discrepancies_found += 1
                rat_discrepancies += 1
                day_has_discrepancy = True
            
            if missing_in_model:
                print(f"    Day {day_num}: {len(missing_in_model)} trials in VTE but not in model: {sorted(missing_in_model)}")
                missing_in_model += len(missing_in_model)
            
            if missing_in_vte_data:
                print(f"    Day {day_num}: {len(missing_in_vte_data)} trials in model but not in VTE: {sorted(missing_in_vte_data)}")
                missing_in_vte += len(missing_in_vte_data)
            
            # Check VTE classification matches for common trials
            vte_mismatches = []
            for trial_id in common_trials:
                vte_classification = vte_data[trial_id]['vte']
                model_classification = model_data[trial_id]['vte']
                
                if vte_classification != model_classification:
                    vte_mismatches.append((trial_id, vte_classification, model_classification))
            
            if vte_mismatches:
                print(f"    Day {day_num}: {len(vte_mismatches)} VTE classification mismatches:")
                for trial_id, vte_val, model_val in vte_mismatches[:5]:  # Show first 5
                    print(f"      Trial {trial_id}: VTE={vte_val}, Model={model_val}")
                if len(vte_mismatches) > 5:
                    print(f"      ... and {len(vte_mismatches) - 5} more")
                vte_classification_mismatches += len(vte_mismatches)
                day_has_discrepancy = True
            
            if not day_has_discrepancy:
                print(f"    Day {day_num}: ✅ {len(common_trials)} trials match perfectly")
        
        if rat_discrepancies == 0:
            print(f"  ✅ All days for {rat} match perfectly")
        print()
    
    # Print summary
    print("\nSUMMARY")
    print("=" * 30)
    print(f"Total rat/day combinations analyzed: {total_comparisons}")
    print(f"Trial count discrepancies: {discrepancies_found}")
    print(f"VTE classification mismatches: {vte_classification_mismatches}")
    print(f"Trials missing in VTE data: {missing_in_vte}")
    print(f"Trials missing in model data: {missing_in_model}")
    
    # Plot VTE distribution by trial type
    plot_vte_by_trial_type(trial_type_vte_counts, title_suffix="(from data_for_model)", save_suffix="model_data")
    plot_vte_by_trial_type(trial_type_vte_counts_zidphi, title_suffix="(from zIdPhi files - boolean VTE)", save_suffix="zidphi_data_boolean")
    plot_vte_by_trial_type(plot_vtes_style_data['trial_type_counts'], title_suffix="(plot_vtes_trial_type.py method)", save_suffix="plot_vtes_style")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON OF VTE DETECTION METHODS")
    print("="*60)
    
    print("\nplot_vtes_trial_type.py raw counts:")
    for trial_type, count in plot_vtes_style_data['VTE_trials'].items():
        total = plot_vtes_style_data['all_trials'].get(trial_type, 0)
        percentage = 100 * count / total if total > 0 else 0
        trial_labels = {1: "AB", 2: "BC", 3: "CD", 4: "DE"}
        label = trial_labels.get(trial_type, str(trial_type))
        print(f"  {label}: {count}/{total} VTEs ({percentage:.1f}%)")
    
    return all_model_data, trial_type_vte_counts, all_vte_data, trial_type_vte_counts_zidphi, plot_vtes_style_data

def plot_vte_by_trial_type(trial_type_data, title_suffix="", save_suffix=""):
    """Plot VTE distribution by trial type"""
    if not trial_type_data:
        print(f"No trial type data available for plotting {title_suffix}")
        return
    
    # Prepare data for plotting
    trial_types = sorted(trial_type_data.keys())
    vte_counts = [trial_type_data[tt]['vte_true'] for tt in trial_types]
    total_counts = [trial_type_data[tt]['total'] for tt in trial_types]
    vte_percentages = [100 * vte_counts[i] / total_counts[i] if total_counts[i] > 0 else 0 
                       for i in range(len(trial_types))]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: VTE counts by trial type
    bars1 = ax1.bar(trial_types, vte_counts, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Trial Type')
    ax1.set_ylabel('Number of VTEs')
    ax1.set_title(f'VTE Counts by Trial Type\n{title_suffix}')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count, total in zip(bars1, vte_counts, total_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}/{total}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: VTE percentages by trial type
    bars2 = ax2.bar(trial_types, vte_percentages, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Trial Type')
    ax2.set_ylabel('VTE Percentage (%)')
    ax2.set_title(f'VTE Percentage by Trial Type\n{title_suffix}')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars2, vte_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = f"/Users/catpillow/Documents/VTE_Analysis/vte_by_trial_type_{save_suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVTE by Trial Type Plot saved to: {save_path}")
    
    # Print trial type summary
    print(f"\nTrial Type Summary {title_suffix}:")
    print("=" * 40)
    for trial_type in trial_types:
        data = trial_type_data[trial_type]
        percentage = 100 * data['vte_true'] / data['total'] if data['total'] > 0 else 0
        print(f"{trial_type:>8}: {data['vte_true']:>4}/{data['total']:>4} VTEs ({percentage:>5.1f}%)")

if __name__ == "__main__":
    try:
        model_data, model_trial_type_data, vte_data, vte_trial_type_data, plot_vtes_data = compare_datasets()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()