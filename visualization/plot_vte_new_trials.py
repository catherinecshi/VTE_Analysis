#!/usr/bin/env python3
"""
VTE Analysis for Newly Introduced Trial Types
============================================

This script analyzes the proportion of VTEs for newly introduced trial types
on the day they are introduced vs their overall average proportion across all days.

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
from scipy.stats import ttest_rel
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
    
    print(f"Training rats: {len(training_rats)} - {training_rats}")
    print(f"Testing rats: {len(testing_rats)} - {testing_rats}")
    print(f"Common rats: {len(common_rats)} - {common_rats}")
    
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
        # Ensure VTE column exists and is boolean
        if 'VTE' in df.columns:
            df['VTE'] = df['VTE'].astype(bool)
        else:
            print(f"Warning: No VTE column in {csv_file}")
            return None
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None

def load_testing_data(rat_name):
    """Load testing data for a rat."""
    testing_file = TESTING_PATH / rat_name / f"{rat_name}_Day1_zIdPhi.csv"
    
    if not testing_file.exists():
        print(f"Testing file not found: {testing_file}")
        return None
    
    try:
        df = pd.read_csv(testing_file)
        
        # Fix column naming inconsistencies
        if 'Trial Type' in df.columns:
            df['Trial_Type'] = df['Trial Type']
        
        if 'VTE' in df.columns:
            df['VTE'] = df['VTE'].astype(bool)
        else:
            print(f"Warning: No VTE column in {testing_file}")
            return None
        return df
    except Exception as e:
        print(f"Error loading {testing_file}: {e}")
        return None

def analyze_rat_training(rat_name):
    """Analyze training data for a single rat to find new trial type introductions."""
    rat_path = BASE_PATH / rat_name
    day_dirs = get_day_directories(rat_path)
    
    if len(day_dirs) <= 1:  # Need at least 2 days (excluding Day 1)
        print(f"Not enough training days for {rat_name}")
        return []
    
    # Track all trial types we've ever seen (including Day 1)
    ever_seen_trial_types = set()
    all_data = []  # Store all training data for overall calculations
    new_trial_introductions = []
    
    for day_dir in day_dirs:
        day_num = int(day_dir.name.replace('Day', ''))
        day_data = load_day_data(day_dir)
        
        if day_data is None:
            continue
            
        # Store all data for later overall calculations
        day_data['Day'] = day_num
        all_data.append(day_data)
        
        current_trial_types = set(day_data['Trial_Type'].unique())
        
        # For Day 1, just record what we've seen but don't count as introductions
        if day_num == 1 or day_num == 2:
            ever_seen_trial_types.update(current_trial_types)
            continue
        
        # Find trial types that are truly new (never seen before, including Day 1)
        new_trial_types = current_trial_types - ever_seen_trial_types
        
        for trial_type in new_trial_types:
            # Calculate VTE proportion for this trial type on first introduction day
            trial_data = day_data[day_data['Trial_Type'] == trial_type]
            if len(trial_data) > 0:
                vte_prop_intro = trial_data['VTE'].mean()
                
                new_trial_introductions.append({
                    'rat': rat_name,
                    'trial_type': trial_type,
                    'introduction_day': day_num,
                    'vte_prop_introduction': vte_prop_intro,
                    'n_trials_introduction': len(trial_data),
                    'phase': 'training'
                })
                
                print(f"    New trial type {trial_type} introduced on Day {day_num}")
        
        # Update the set of all trial types we've ever seen
        ever_seen_trial_types.update(current_trial_types)
    
    # Calculate overall averages for each trial type
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        for intro in new_trial_introductions:
            trial_type = intro['trial_type']
            trial_data = combined_data[combined_data['Trial_Type'] == trial_type]
            intro['vte_prop_overall'] = trial_data['VTE'].mean()
            intro['n_trials_overall'] = len(trial_data)
    
    return new_trial_introductions, ever_seen_trial_types

def analyze_rat_testing(rat_name, training_trial_types):
    """Analyze testing data for novel trial types."""
    testing_data = load_testing_data(rat_name)
    
    if testing_data is None:
        return []
    
    # Find trial types in testing that weren't in training
    testing_trial_types = set(testing_data['Trial_Type'].unique())
    novel_trial_types = testing_trial_types - training_trial_types
    
    novel_introductions = []
    
    for trial_type in novel_trial_types:
        trial_data = testing_data[testing_data['Trial_Type'] == trial_type]
        if len(trial_data) > 0:
            vte_prop = trial_data['VTE'].mean()
            
            novel_introductions.append({
                'rat': rat_name,
                'trial_type': trial_type,
                'introduction_day': 'Testing',
                'vte_prop_introduction': vte_prop,  # Same as overall for testing
                'vte_prop_overall': vte_prop,
                'n_trials_introduction': len(trial_data),
                'n_trials_overall': len(trial_data),
                'phase': 'testing'
            })
    
    return novel_introductions

def main():
    """Main analysis function."""
    print("=== VTE Analysis for Newly Introduced Trial Types ===\n")
    
    # Step 1: Get rats with both training and testing data
    print("Step 1: Identifying rats with both training and testing data...")
    common_rats = get_rats_with_both_data()
    
    if not common_rats:
        print("No rats found with both training and testing data!")
        return
    
    print(f"\nAnalyzing {len(common_rats)} rats: {common_rats}\n")
    
    # Step 2: Analyze each rat
    all_results = []
    
    for rat_name in common_rats:
        print(f"Analyzing {rat_name}...")
        
        if rat_name == "BP06" or rat_name == "BP07" or rat_name == "BP08":
            continue
        
        # Training analysis
        try:
            training_results, training_trial_types = analyze_rat_training(rat_name)
            print(f"  Training: Found {len(training_results)} new trial type introductions")
            
            # Testing analysis
            testing_results = analyze_rat_testing(rat_name, training_trial_types)
            print(f"  Testing: Found {len(testing_results)} novel trial types")
            
            # Combine results
            all_results.extend(training_results)
            all_results.extend(testing_results)
            
        except Exception as e:
            print(f"  Error analyzing {rat_name}: {e}")
            continue
    
    if not all_results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    print(f"\nTotal analysis records: {len(results_df)}")
    print(f"Training records: {len(results_df[results_df['phase'] == 'training'])}")
    print(f"Testing records: {len(results_df[results_df['phase'] == 'testing'])}")
    
    # Save intermediate results
    results_df.to_csv('/Users/catpillow/Documents/VTE_Analysis/vte_new_trial_results.csv', index=False)
    print("\nSaved results to vte_new_trial_results.csv")
    
    # Step 3: Statistical analysis and visualization
    perform_statistical_analysis(results_df)
    
    return results_df

def calculate_effect_size(x, y):
    """Calculate Cohen's d effect size."""
    diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
    return diff / pooled_std if pooled_std > 0 else 0

def perform_statistical_analysis(results_df):
    """Perform statistical analysis and create visualization."""
    print("\n=== Statistical Analysis ===")
    
    # Separate training and testing data
    training_data = results_df[results_df['phase'] == 'training'].copy()
    testing_data = results_df[results_df['phase'] == 'testing'].copy()
    
    print(f"Training data: {len(training_data)} records from {training_data['rat'].nunique()} rats")
    print(f"Testing data: {len(testing_data)} records from {testing_data['rat'].nunique()} rats")
    
    # Statistical analysis for training data
    if len(training_data) > 0:
        intro_vals = training_data['vte_prop_introduction'].values
        overall_vals = training_data['vte_prop_overall'].values
        
        # Paired t-test
        t_stat, p_value = ttest_rel(intro_vals, overall_vals)
        effect_size = calculate_effect_size(intro_vals, overall_vals)
        
        print(f"\nTraining Phase Analysis:")
        print(f"  Introduction day VTE proportion: {np.mean(intro_vals):.4f} ± {np.std(intro_vals, ddof=1):.4f}")
        print(f"  Overall average VTE proportion: {np.mean(overall_vals):.4f} ± {np.std(overall_vals, ddof=1):.4f}")
        print(f"  Paired t-test: t({len(intro_vals)-1}) = {t_stat:.4f}, p = {p_value:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        
        # Interpretation
        if p_value < 0.05:
            direction = "higher" if np.mean(intro_vals) > np.mean(overall_vals) else "lower"
            print(f"  ** Significant difference: VTE rates are {direction} on introduction days **")
        else:
            print(f"  No significant difference between introduction day and overall VTE rates")
    
    # For testing data, introduction = overall, so we just report descriptive statistics
    if len(testing_data) > 0:
        vte_vals = testing_data['vte_prop_introduction'].values
        print(f"\nTesting Phase Analysis (novel trial types):")
        print(f"  VTE proportion for novel trial types: {np.mean(vte_vals):.4f} ± {np.std(vte_vals, ddof=1):.4f}")
        print(f"  Range: {np.min(vte_vals):.4f} to {np.max(vte_vals):.4f}")
    
    # Create visualization
    create_visualization(results_df, training_data, testing_data)

def create_visualization(results_df, training_data, testing_data):
    """Create the final visualization."""
    print("\n=== Creating Visualization ===")
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training phase plot
    if len(training_data) > 0:
        # Prepare data for plotting
        intro_vals = training_data['vte_prop_introduction'].values
        overall_vals = training_data['vte_prop_overall'].values
        
        # Calculate means and SEM
        intro_mean = np.mean(intro_vals)
        intro_sem = np.std(intro_vals, ddof=1) / np.sqrt(len(intro_vals))
        overall_mean = np.mean(overall_vals)
        overall_sem = np.std(overall_vals, ddof=1) / np.sqrt(len(overall_vals))
        
        # Bar plot
        x_pos = np.array([0, 1])
        means = [intro_mean, overall_mean]
        sems = [intro_sem, overall_sem]
        
        bars = ax1.bar(x_pos, means, yerr=sems, capsize=5, 
                       color=['#FF6B6B', '#4ECDC4'], alpha=0.8,
                       edgecolor='black', linewidth=1)
        
        # Add individual data points
        np.random.seed(42)  # For reproducible jitter
        jitter1 = np.random.normal(0, 0.05, len(intro_vals))
        jitter2 = np.random.normal(1, 0.05, len(overall_vals))
        
        ax1.scatter(jitter1, intro_vals, alpha=0.6, s=30, color='darkred', zorder=3)
        ax1.scatter(jitter2, overall_vals, alpha=0.6, s=30, color='darkgreen', zorder=3)
        
        # Connect paired points
        for i in range(len(intro_vals)):
            ax1.plot([jitter1[i], jitter2[i]], [intro_vals[i], overall_vals[i]], 
                    'k-', alpha=0.3, linewidth=0.5)
        
        # Formatting
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Introduction Day', 'Overall Average'])
        ax1.set_ylabel('VTE Proportion')
        ax1.set_title('Training Phase: VTE Rates for New Trial Types\n(n={} trial type introductions)'.format(len(intro_vals)))
        ax1.set_ylim(0, max(max(intro_vals), max(overall_vals)) * 1.1)
        
        # Add statistical annotation
        t_stat, p_value = ttest_rel(intro_vals, overall_vals)
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Add significance line
        y_max = max(max(intro_vals), max(overall_vals))
        ax1.plot([0, 1], [y_max*1.05, y_max*1.05], 'k-', linewidth=1)
        ax1.text(0.5, y_max*1.07, f'p = {p_value:.3f} {sig_text}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Testing phase plot
    if len(testing_data) > 0:
        # Group by rat and trial type for testing data
        testing_summary = testing_data.groupby('rat')['vte_prop_introduction'].agg(['mean', 'count']).reset_index()
        
        vte_vals = testing_summary['mean'].values
        
        # Simple bar plot for testing data
        ax2.bar([0], [np.mean(vte_vals)], 
                yerr=[np.std(vte_vals, ddof=1) / np.sqrt(len(vte_vals))],
                capsize=5, color='#9B59B6', alpha=0.8,
                edgecolor='black', linewidth=1)
        
        # Add individual rat data points
        jitter = np.random.normal(0, 0.05, len(vte_vals))
        ax2.scatter(jitter, vte_vals, alpha=0.6, s=30, color='purple', zorder=3)
        
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Novel Trial Types'])
        ax2.set_ylabel('VTE Proportion')
        ax2.set_title('Testing Phase: VTE Rates for Novel Trial Types\n(n={} rats)'.format(len(vte_vals)))
        ax2.set_ylim(0, max(vte_vals) * 1.2 if len(vte_vals) > 0 else 1)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('/Users/catpillow/Documents/VTE_Analysis/vte_new_trial_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("Saved plot to vte_new_trial_analysis.png")
    
    plt.show()

if __name__ == "__main__":
    results_df = main()