#!/usr/bin/env python3
"""
VTE Analysis for Newly Introduced Trial Types: Correct vs Incorrect Performance
============================================================================

This script analyzes VTE rates for newly introduced trial types,
comparing VTE proportions between correct and incorrect trials.

Author: Generated for VTE Analysis Project
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_PATH = Path("/Users/catpillow/Documents/VTE_Analysis/processed_data/VTE_values")
TESTING_PATH = BASE_PATH / "inferenceTesting"

def get_rats_with_both_data():
    """Identify rats that have both training and testing data."""
    # Get training rats
    training_rats = [d.name for d in BASE_PATH.iterdir() 
                    if d.is_dir() and d.name != "inferenceTesting" and 
                    re.match(r'^[A-Z]{2}[0-9]+$|^TH[0-9]+$', d.name)]
    
    # Get testing rats  
    testing_rats = [d.name for d in TESTING_PATH.iterdir() if d.is_dir()]
    
    # Find intersection
    common_rats = list(set(training_rats) & set(testing_rats))
    common_rats.sort()
    
    return common_rats

def get_day_directories(rat_path):
    """Get all day directories for a rat, sorted numerically."""
    day_dirs = []
    for d in rat_path.iterdir():
        if d.is_dir() and d.name.startswith('Day'):
            try:
                day_num = int(d.name.replace('Day', ''))
                day_dirs.append((day_num, d))
            except ValueError:
                continue
    
    # Sort by day number and return directories
    day_dirs.sort(key=lambda x: x[0])
    return [d[1] for d in day_dirs]

def load_day_data(day_dir):
    """Load VTE data for a specific day."""
    day_name = day_dir.name
    csv_file = day_dir / f"zIdPhi_day_{day_name}.csv"
    
    if not csv_file.exists():
        return None
    
    try:
        df = pd.read_csv(csv_file)
        # Ensure required columns exist and are proper type
        if 'VTE' in df.columns and 'Correct' in df.columns:
            df['VTE'] = df['VTE'].astype(bool)
            df['Correct'] = df['Correct'].astype(bool)
        else:
            print(f"Warning: Missing VTE or Correct column in {csv_file}")
            return None
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None

def load_testing_data(rat_name):
    """Load testing data for a rat."""
    testing_file = TESTING_PATH / rat_name / f"{rat_name}_Day1_zIdPhi.csv"
    
    if not testing_file.exists():
        return None
    
    try:
        df = pd.read_csv(testing_file)
        
        # Fix column naming inconsistencies
        if 'Trial Type' in df.columns:
            df['Trial_Type'] = df['Trial Type']
        
        if 'VTE' in df.columns and 'Correct' in df.columns:
            df['VTE'] = df['VTE'].astype(bool)
            df['Correct'] = df['Correct'].astype(bool)
        else:
            print(f"Warning: Missing VTE or Correct column in {testing_file}")
            return None
        return df
    except Exception as e:
        print(f"Error loading {testing_file}: {e}")
        return None

def find_new_trial_introductions(rat_name):
    """Find all new trial type introductions for a rat."""
    rat_path = BASE_PATH / rat_name
    day_dirs = get_day_directories(rat_path)
    
    if len(day_dirs) <= 1:
        return []
    
    # Track all trial types we've ever seen
    ever_seen_trial_types = set()
    new_trial_data = []
    
    for day_dir in day_dirs:
        day_num = int(day_dir.name.replace('Day', ''))
        day_data = load_day_data(day_dir)
        
        if day_data is None:
            continue
        
        current_trial_types = set(day_data['Trial_Type'].unique())
        
        # For Day 1 and 2, just record what we've seen but don't count as introductions
        if day_num <= 2:
            ever_seen_trial_types.update(current_trial_types)
            continue
        
        # Find trial types that are truly new
        new_trial_types = current_trial_types - ever_seen_trial_types
        
        for trial_type in new_trial_types:
            # Get all trials for this trial type on introduction day
            trial_data = day_data[day_data['Trial_Type'] == trial_type].copy()
            if len(trial_data) > 0:
                trial_data['rat'] = rat_name
                trial_data['introduction_day'] = day_num
                trial_data['phase'] = 'training'
                new_trial_data.append(trial_data)
        
        # Update seen trial types
        ever_seen_trial_types.update(current_trial_types)
    
    return new_trial_data

def find_novel_testing_trials(rat_name, training_trial_types):
    """Find novel trial types in testing data."""
    testing_data = load_testing_data(rat_name)
    
    if testing_data is None:
        return []
    
    # Find trial types in testing that weren't in training
    testing_trial_types = set(testing_data['Trial_Type'].unique())
    novel_trial_types = testing_trial_types - training_trial_types
    
    novel_trial_data = []
    
    for trial_type in novel_trial_types:
        trial_data = testing_data[testing_data['Trial_Type'] == trial_type].copy()
        if len(trial_data) > 0:
            trial_data['rat'] = rat_name
            trial_data['introduction_day'] = 'Testing'
            trial_data['phase'] = 'testing'
            novel_trial_data.append(trial_data)
    
    return novel_trial_data

def analyze_vte_by_correctness(all_trial_data):
    """Analyze VTE rates by correct/incorrect performance."""
    
    # Separate correct and incorrect trials
    correct_trials = all_trial_data[all_trial_data['Correct'] == True]
    incorrect_trials = all_trial_data[all_trial_data['Correct'] == False]
    
    print(f"Total trials: {len(all_trial_data)}")
    print(f"Correct trials: {len(correct_trials)} ({len(correct_trials)/len(all_trial_data)*100:.1f}%)")
    print(f"Incorrect trials: {len(incorrect_trials)} ({len(incorrect_trials)/len(all_trial_data)*100:.1f}%)")
    
    # Calculate VTE rates
    correct_vte_rate = correct_trials['VTE'].mean() if len(correct_trials) > 0 else 0
    incorrect_vte_rate = incorrect_trials['VTE'].mean() if len(incorrect_trials) > 0 else 0
    
    print(f"VTE rate for correct trials: {correct_vte_rate:.4f}")
    print(f"VTE rate for incorrect trials: {incorrect_vte_rate:.4f}")
    
    # Statistical test
    if len(correct_trials) > 0 and len(incorrect_trials) > 0:
        # Chi-square test for independence
        contingency_table = pd.crosstab(all_trial_data['Correct'], all_trial_data['VTE'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nContingency table:")
        print(contingency_table)
        print(f"Chi-square test: χ²({dof}) = {chi2:.4f}, p = {p_value:.4f}")
        
        # Independent t-test on VTE rates
        correct_vtes = correct_trials['VTE'].astype(int)
        incorrect_vtes = incorrect_trials['VTE'].astype(int)
        t_stat, t_p = ttest_ind(correct_vtes, incorrect_vtes)
        print(f"Independent t-test: t = {t_stat:.4f}, p = {t_p:.4f}")
    
    return correct_vte_rate, incorrect_vte_rate, correct_trials, incorrect_trials

def create_visualization(all_trial_data, training_data, testing_data):
    """Create visualization of VTE rates by correctness."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall analysis
    ax1 = axes[0, 0]
    correct_rate = all_trial_data[all_trial_data['Correct'] == True]['VTE'].mean()
    incorrect_rate = all_trial_data[all_trial_data['Correct'] == False]['VTE'].mean()
    
    bars = ax1.bar(['Correct', 'Incorrect'], [correct_rate, incorrect_rate], 
                   color=['#2ECC71', '#E74C3C'], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('VTE Proportion')
    ax1.set_title('Overall: VTE Rates by Trial Correctness\n(Newly Introduced Trial Types)')
    ax1.set_ylim(0, max(correct_rate, incorrect_rate) * 1.2)
    
    # Add value labels on bars
    for bar, value in zip(bars, [correct_rate, incorrect_rate]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training phase
    if len(training_data) > 0:
        ax2 = axes[0, 1]
        train_correct_rate = training_data[training_data['Correct'] == True]['VTE'].mean()
        train_incorrect_rate = training_data[training_data['Correct'] == False]['VTE'].mean()
        
        bars = ax2.bar(['Correct', 'Incorrect'], [train_correct_rate, train_incorrect_rate], 
                       color=['#2ECC71', '#E74C3C'], alpha=0.8, edgecolor='black')
        ax2.set_ylabel('VTE Proportion')
        ax2.set_title('Training Phase: VTE Rates by Trial Correctness')
        ax2.set_ylim(0, max(train_correct_rate, train_incorrect_rate) * 1.2)
        
        for bar, value in zip(bars, [train_correct_rate, train_incorrect_rate]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Testing phase
    if len(testing_data) > 0:
        ax3 = axes[1, 0]
        test_correct_rate = testing_data[testing_data['Correct'] == True]['VTE'].mean()
        test_incorrect_rate = testing_data[testing_data['Correct'] == False]['VTE'].mean()
        
        bars = ax3.bar(['Correct', 'Incorrect'], [test_correct_rate, test_incorrect_rate], 
                       color=['#2ECC71', '#E74C3C'], alpha=0.8, edgecolor='black')
        ax3.set_ylabel('VTE Proportion')
        ax3.set_title('Testing Phase: VTE Rates by Trial Correctness')
        ax3.set_ylim(0, max(test_correct_rate, test_incorrect_rate) * 1.2)
        
        for bar, value in zip(bars, [test_correct_rate, test_incorrect_rate]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Individual rat breakdown
    ax4 = axes[1, 1]
    
    # Calculate VTE rates by rat and correctness
    rat_summary = []
    for rat in all_trial_data['rat'].unique():
        rat_data = all_trial_data[all_trial_data['rat'] == rat]
        correct_data = rat_data[rat_data['Correct'] == True]
        incorrect_data = rat_data[rat_data['Correct'] == False]
        
        if len(correct_data) > 0:
            rat_summary.append({
                'rat': rat,
                'correctness': 'Correct',
                'vte_rate': correct_data['VTE'].mean(),
                'n_trials': len(correct_data)
            })
        
        if len(incorrect_data) > 0:
            rat_summary.append({
                'rat': rat,
                'correctness': 'Incorrect', 
                'vte_rate': incorrect_data['VTE'].mean(),
                'n_trials': len(incorrect_data)
            })
    
    if rat_summary:
        rat_df = pd.DataFrame(rat_summary)
        
        # Box plot by correctness
        sns.boxplot(data=rat_df, x='correctness', y='vte_rate', ax=ax4)
        sns.stripplot(data=rat_df, x='correctness', y='vte_rate', ax=ax4, 
                     color='black', alpha=0.6, size=4)
        
        ax4.set_ylabel('VTE Proportion')
        ax4.set_xlabel('Trial Correctness')
        ax4.set_title('Individual Rat VTE Rates by Correctness')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/Users/catpillow/Documents/VTE_Analysis/vte_correct_incorrect_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("\nSaved plot to vte_correct_incorrect_analysis.png")
    
    plt.show()

def main():
    """Main analysis function."""
    print("=== VTE Analysis: Correct vs Incorrect Trials for New Trial Types ===\n")
    
    # Get rats with both training and testing data
    common_rats = get_rats_with_both_data()
    
    # Filter out specific rats as requested
    common_rats = [rat for rat in common_rats if rat not in ["BP06", "BP07", "BP08"]]
    
    print(f"Analyzing {len(common_rats)} rats: {common_rats}\n")
    
    all_trial_data = []
    
    for rat_name in common_rats:
        print(f"Analyzing {rat_name}...")
        
        # Get training trial types first
        rat_path = BASE_PATH / rat_name
        day_dirs = get_day_directories(rat_path)
        training_trial_types = set()
        
        for day_dir in day_dirs:
            day_num = int(day_dir.name.replace('Day', ''))
            if day_num <= 2:  # Day 1 and 2
                day_data = load_day_data(day_dir)
                if day_data is not None:
                    training_trial_types.update(day_data['Trial_Type'].unique())
        
        # Find new trial introductions in training
        training_introductions = find_new_trial_introductions(rat_name)
        
        # Find novel trials in testing
        testing_introductions = find_novel_testing_trials(rat_name, training_trial_types)
        
        # Combine all data
        rat_data = training_introductions + testing_introductions
        if rat_data:
            combined_data = pd.concat(rat_data, ignore_index=True)
            all_trial_data.append(combined_data)
            
            print(f"  Found {len(combined_data)} trials from new trial type introductions")
    
    if not all_trial_data:
        print("No data found!")
        return
    
    # Combine all rat data
    all_data = pd.concat(all_trial_data, ignore_index=True)
    training_data = all_data[all_data['phase'] == 'training']
    testing_data = all_data[all_data['phase'] == 'testing']
    
    print(f"\n=== Overall Results ===")
    correct_rate, incorrect_rate, correct_trials, incorrect_trials = analyze_vte_by_correctness(all_data)
    
    print(f"\n=== Training Phase Results ===")
    if len(training_data) > 0:
        analyze_vte_by_correctness(training_data)
    
    print(f"\n=== Testing Phase Results ===")
    if len(testing_data) > 0:
        analyze_vte_by_correctness(testing_data)
    
    # Create visualization
    create_visualization(all_data, training_data, testing_data)
    
    # Save detailed results
    all_data.to_csv('/Users/catpillow/Documents/VTE_Analysis/vte_correct_incorrect_data.csv', index=False)
    print("Saved detailed data to vte_correct_incorrect_data.csv")

if __name__ == "__main__":
    main()