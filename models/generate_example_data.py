#!/usr/bin/env python3
"""
Generate Example Data for Betasort Analysis Testing

This script creates synthetic data matching the format expected by the betasort analysis pipeline.
It generates:
- 10 training days with progressive trial type introduction
- 1 transitive inference testing day
- All data follows the hierarchy: A(0) > B(1) > C(2) > D(3) > E(4)

Usage:
    python generate_example_data.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(43)

def create_trial(trial_type, correct_probability=0.8):
    """
    Create a single trial based on trial type.
    
    Parameters:
    -----------
    trial_type : str
        One of 'AB', 'BC', 'CD', 'DE', 'BD', 'AE'
    correct_probability : float
        Probability that the trial will be correct (default 0.8)
    
    Returns:
    --------
    dict : Trial data with 'first', 'second', 'correct'
    """
    # Define stimulus mappings
    stimulus_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # Get the two stimuli for this trial type
    stim1_letter = trial_type[0]
    stim2_letter = trial_type[1]
    stim1 = stimulus_map[stim1_letter]
    stim2 = stimulus_map[stim2_letter]
    
    # Determine if this trial should be correct
    is_correct = np.random.random() < correct_probability
    
    if is_correct:
        # Correct: lower stimulus index should be first (higher in hierarchy)
        first, second = min(stim1, stim2), max(stim1, stim2)
        correct = 1
    else:
        # Incorrect: higher stimulus index should be first (lower in hierarchy)
        first, second = max(stim1, stim2), min(stim1, stim2)
        correct = 0
    
    return {
        'first': first,
        'second': second, 
        'correct': correct,
        'VTE': False,  # All VTE are False as specified
        'length': 0    # All lengths are 0 as specified
    }

def generate_day_data(day, trial_types, n_trials=100):
    """
    Generate data for a single day.
    
    Parameters:
    -----------
    day : int
        Day number
    trial_types : list
        List of trial types available for this day
    n_trials : int
        Number of trials per day (default 100)
    
    Returns:
    --------
    pd.DataFrame : Day's trial data
    """
    trials = []
    
    for trial_id in range(1, n_trials + 1):
        # Randomly select trial type with equal probability
        trial_type = np.random.choice(trial_types)
        
        # Create the trial
        trial = create_trial(trial_type)
        trial['ID'] = trial_id
        trial['Day'] = day
        
        trials.append(trial)
    
    # Create DataFrame with correct column order
    df = pd.DataFrame(trials)
    df = df[['ID', 'Day', 'first', 'second', 'correct', 'VTE', 'length']]
    
    return df

def generate_inference_testing_data(n_trials=100):
    """
    Generate transitive inference testing data.
    
    Parameters:
    -----------
    n_trials : int
        Number of trials (default 100)
    
    Returns:
    --------
    pd.DataFrame : Inference testing trial data
    """
    trial_types = ['AB', 'BC', 'CD', 'DE', 'BD', 'AE']
    trials = []
    
    for trial_id in range(1, n_trials + 1):
        # Randomly select trial type with equal probability
        trial_type = np.random.choice(trial_types)
        
        # Create the trial
        trial = create_trial(trial_type)
        trial['ID'] = trial_id
        
        trials.append(trial)
    
    # Create DataFrame with correct column order (no Day column for inference testing)
    df = pd.DataFrame(trials)
    df = df[['ID', 'first', 'second', 'correct', 'VTE', 'length']]
    
    return df

def generate_single_rat_data(rat_name, base_path, seed=None):
    """
    Generate data for a single example rat.
    
    Parameters:
    -----------
    rat_name : str
        Name of the rat (e.g., "ExampleRat1")
    base_path : Path
        Base path for data storage
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    example_rat_path = base_path / rat_name
    inference_path = base_path / "inferenceTesting" / rat_name
    
    # Create directories
    example_rat_path.mkdir(parents=True, exist_ok=True)
    inference_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating example data for {rat_name} (seed={seed})...")
    
    # Define trial progression schedule
    day_schedules = {
        1: ['AB', 'BC'],           # Days 1-7: AB and BC only
        2: ['AB', 'BC'],
        3: ['AB', 'BC'],
        4: ['AB', 'BC'],
        5: ['AB', 'BC'],
        6: ['AB', 'BC'],
        7: ['AB', 'BC'],
        8: ['AB', 'BC', 'CD'],     # Days 8-13: AB, BC, and CD
        9: ['AB', 'BC', 'CD'],
        10: ['AB', 'BC', 'CD'],
        11: ['AB', 'BC', 'CD'],
        12: ['AB', 'BC', 'CD'],
        13: ['AB', 'BC', 'CD'],
        14: ['AB', 'BC', 'CD', 'DE'],  # Days 14-24: AB, BC, CD, and DE
        15: ['AB', 'BC', 'CD', 'DE'],
        16: ['AB', 'BC', 'CD', 'DE'],
        17: ['AB', 'BC', 'CD', 'DE'],
        18: ['AB', 'BC', 'CD', 'DE'],
        19: ['AB', 'BC', 'CD', 'DE'],
        20: ['AB', 'BC', 'CD', 'DE'],
        21: ['AB', 'BC', 'CD', 'DE'],
        22: ['AB', 'BC', 'CD', 'DE'],
        23: ['AB', 'BC', 'CD', 'DE'],
        24: ['AB', 'BC', 'CD', 'DE']
    }
    
    # Generate training days
    all_data = []
    for day in range(1, 25):
        day_data = generate_day_data(day, day_schedules[day])
        all_data.append(day_data)
        
        # Save individual day data
        day_filename = example_rat_path / f"Day{day}.csv"
        day_data.to_csv(day_filename, index=False)
    
    # Create combined all_days file
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_filename = example_rat_path / f"{rat_name}_all_days.csv"
    combined_df.to_csv(combined_filename, index=False)
    
    # Generate inference testing data
    inference_data = generate_inference_testing_data()
    inference_filename = inference_path / f"{rat_name}.csv"
    inference_data.to_csv(inference_filename, index=False)
    
    # Calculate summary statistics
    total_trials = len(combined_df)
    correct_trials = combined_df['correct'].sum()
    correct_rate = correct_trials / total_trials
    
    inference_total = len(inference_data)
    inference_correct = inference_data['correct'].sum()
    inference_rate = inference_correct / inference_total
    
    print(f"  Training: {total_trials} trials across 24 days, {correct_rate:.1%} correct")
    print(f"  Inference: {inference_total} trials, {inference_rate:.1%} correct")
    
    return {
        'rat_name': rat_name,
        'training_trials': total_trials,
        'training_correct_rate': correct_rate,
        'inference_trials': inference_total,
        'inference_correct_rate': inference_rate,
        'seed': seed
    }

def main():
    """
    Main function to generate all example data.
    """
    # Define base paths
    base_path = Path("/Users/catpillow/Documents/VTE_Analysis/processed_data/data_for_model")
    
    # Generate 10 different example rats with different seeds
    n_rats = 10
    summary_stats = []
    
    print(f"Generating {n_rats} example rats with different random seeds...")
    print("=" * 60)
    
    for i in range(1, n_rats + 1):
        rat_name = f"ExampleRat{i}"
        seed = 42 + i  # Different seed for each rat
        
        stats = generate_single_rat_data(rat_name, base_path, seed)
        summary_stats.append(stats)
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY OF GENERATED EXAMPLE RATS:")
    print("=" * 60)
    print(f"{'Rat Name':<12} {'Seed':<6} {'Training %':<12} {'Inference %':<12}")
    print("-" * 60)
    
    for stats in summary_stats:
        print(f"{stats['rat_name']:<12} {stats['seed']:<6} "
              f"{stats['training_correct_rate']:<11.1%} {stats['inference_correct_rate']:<11.1%}")
    
    avg_training = np.mean([s['training_correct_rate'] for s in summary_stats])
    avg_inference = np.mean([s['inference_correct_rate'] for s in summary_stats])
    
    print("-" * 60)
    print(f"{'Average':<12} {'N/A':<6} {avg_training:<11.1%} {avg_inference:<11.1%}")
    
    print(f"\n✓ Generated {n_rats} example rats successfully!")
    print(f"✓ Training data: processed_data/data_for_model/ExampleRat[1-{n_rats}]/")
    print(f"✓ Inference data: processed_data/data_for_model/inferenceTesting/ExampleRat[1-{n_rats}]/")
    print(f"\nTo analyze all example rats:")
    print(f"  from analysis.betasort_analysis_pipeline import analyze_example_rats")
    print(f"  analyze_example_rats()")


def generate_legacy_single_rat():
    """
    Generate the original single ExampleRat for backward compatibility.
    """
    base_path = Path("/Users/catpillow/Documents/VTE_Analysis/processed_data/data_for_model")
    example_rat_name = "ExampleRat"
    example_rat_path = base_path / example_rat_name
    inference_path = base_path / "inferenceTesting" / example_rat_name
    
    # Create directories
    example_rat_path.mkdir(parents=True, exist_ok=True)
    inference_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating example data for {example_rat_name}...")
    print(f"Training data will be saved to: {example_rat_path}")
    print(f"Inference testing data will be saved to: {inference_path}")
    
    # Define trial progression schedule
    day_schedules = {
        1: ['AB', 'BC'],           # Days 1-7: AB and BC only
        2: ['AB', 'BC'],
        3: ['AB', 'BC'],
        4: ['AB', 'BC'],
        5: ['AB', 'BC'],
        6: ['AB', 'BC'],
        7: ['AB', 'BC'],
        8: ['AB', 'BC', 'CD'],     # Days 8-13: AB, BC, and CD
        9: ['AB', 'BC', 'CD'],
        10: ['AB', 'BC', 'CD'],
        11: ['AB', 'BC', 'CD'],
        12: ['AB', 'BC', 'CD'],
        13: ['AB', 'BC', 'CD'],
        14: ['AB', 'BC', 'CD', 'DE'],  # Days 14-24: AB, BC, CD, and DE
        15: ['AB', 'BC', 'CD', 'DE'],
        16: ['AB', 'BC', 'CD', 'DE'],
        17: ['AB', 'BC', 'CD', 'DE'],
        18: ['AB', 'BC', 'CD', 'DE'],
        19: ['AB', 'BC', 'CD', 'DE'],
        20: ['AB', 'BC', 'CD', 'DE'],
        21: ['AB', 'BC', 'CD', 'DE'],
        22: ['AB', 'BC', 'CD', 'DE'],
        23: ['AB', 'BC', 'CD', 'DE'],
        24: ['AB', 'BC', 'CD', 'DE']
    }
    
    # Generate training days
    print("\nGenerating training days...")
    for day in range(1, 25):
        print(f"  Day {day}: {day_schedules[day]}")
        
        day_data = generate_day_data(day, day_schedules[day])
        
        # Save day data
        day_filename = example_rat_path / f"Day{day}.csv"
        day_data.to_csv(day_filename, index=False)
        
        # Print summary statistics
        total_trials = len(day_data)
        correct_trials = day_data['correct'].sum()
        correct_rate = correct_trials / total_trials
        print(f"    Saved {total_trials} trials, {correct_rate:.1%} correct")
    
    # Generate inference testing data
    print(f"\nGenerating transitive inference testing data...")
    inference_data = generate_inference_testing_data()
    
    # Save inference testing data
    inference_filename = inference_path / f"{example_rat_name}.csv"
    inference_data.to_csv(inference_filename, index=False)
    
    # Print summary statistics
    total_trials = len(inference_data)
    correct_trials = inference_data['correct'].sum()
    correct_rate = correct_trials / total_trials
    print(f"  Saved {total_trials} trials, {correct_rate:.1%} correct")
    
    # Print trial type breakdown for inference testing
    print("\nInference testing trial type breakdown:")
    trial_types = ['AB', 'BC', 'CD', 'DE', 'BD', 'AE']
    for trial_type in trial_types:
        # Count trials for this type
        stim_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        stim1, stim2 = stim_map[trial_type[0]], stim_map[trial_type[1]]
        
        type_trials = inference_data[
            ((inference_data['first'] == stim1) & (inference_data['second'] == stim2)) |
            ((inference_data['first'] == stim2) & (inference_data['second'] == stim1))
        ]
        
        count = len(type_trials)
        if count > 0:
            correct_rate = type_trials['correct'].mean()
            print(f"  {trial_type}: {count} trials ({count/total_trials:.1%}), {correct_rate:.1%} correct")
    
    print(f"\n✓ Example data generation complete!")
    print(f"✓ Training data: {example_rat_path}")
    print(f"✓ Inference testing data: {inference_path}")
    print(f"\nYou can now test this data with the betasort analysis pipeline:")
    print(f"  pipeline = BetasortPipeline(rats_to_include=['{example_rat_name}'])")
    print(f"  pipeline.run_full_analysis()")

if __name__ == "__main__":
    main()