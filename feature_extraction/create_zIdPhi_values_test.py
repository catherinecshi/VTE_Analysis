"""
Modified script to create zIdPhi values for testing data using training data parameters.
This script:
1. Loads training data to establish z-scoring parameters and VTE thresholds
2. Applies those parameters to testing data
3. Saves results in the testing folders
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import zscore

from config.paths import paths
from preprocessing import data_processing
from utilities import logging_utils

### LOGGING
logger = logging_utils.setup_script_logger()

base_path = "/Users/catpillow/Documents/VTE_Analysis"
dlc_path = paths.cleaned_dlc
data_path = paths.vte_data
data_structure = data_processing.load_data_structure(data_path)
vte_path = paths.vertice_data

# Dictionary to store training parameters for each rat
training_params = {}

print("Processing training data to establish parameters...")

# First pass: Process training data to get z-scoring parameters and thresholds
for rat_dir in paths.vte_values.iterdir():
    rat = rat_dir.name
    if ".DS" in rat or ".csv" in rat or rat == "inferenceTesting":  # Skip testing folder
        continue
    
    rat_path = paths.vte_values / rat
    before_zscore_df = pd.DataFrame()
    
    # Collect all training data across days for this rat
    for day_dir in rat_path.iterdir():
        day = day_dir.name
        day_path = rat_path / day
        if ".DS" in day:
            continue
        
        for root, _, files in os.walk(day_path):
            for file in files:
                if "trajectories.csv" not in file:
                    continue
                
                file_path = os.path.join(root, file)
                traj_info = pd.read_csv(file_path, header=0)
                
                if all(col in traj_info.columns for col in ["IdPhi", "Choice", "ID", "Trial Type", "Length"]):
                    rows = {"ID": traj_info["ID"], "Day": day, "Choice": traj_info["Choice"], "Correct": traj_info["Correct"],
                            "Trial_Type": traj_info["Trial Type"], "IdPhi": traj_info["IdPhi"], "Length": traj_info["Length"]}
                    temp_df = pd.DataFrame(rows)
                    before_zscore_df = pd.concat([before_zscore_df, temp_df], ignore_index=True)
                else:
                    logger.error(f"missing columns in {rat} on {day}")
    
    if before_zscore_df.empty:
        logger.warning(f"No training data found for {rat}")
        continue
    
    # Filter by length (same as original)
    try:
        valid_length_df = before_zscore_df[before_zscore_df["Length"] <= 4].copy()
    except KeyError:
        logger.error(f"Length column missing for {rat}")
        continue
    
    # Calculate z-scoring parameters grouped by choice
    try:
        grouped_by_choice = valid_length_df.groupby(by="Choice")
    except Exception as e:
        logger.error(f"error with groupby for {rat} - {e}")
        continue
    
    # Store z-scoring parameters for each choice
    choice_params = {}
    zscored_valid_df = pd.DataFrame()
    
    for choice, choice_group in grouped_by_choice:
        if len(choice_group) > 1:
            # Store mean and std for this choice
            choice_params[choice] = {
                'mean': np.mean(choice_group["IdPhi"]),
                'std': np.std(choice_group["IdPhi"])
            }
            
            # Calculate z-scores for training data
            zIdPhis = zscore(choice_group["IdPhi"])
            choice_group["zIdPhi"] = zIdPhis
            zscored_valid_df = pd.concat([zscored_valid_df, choice_group], ignore_index=True)
        else:
            logger.warning(f"Skipping choice {choice} for {rat} - insufficient samples ({len(choice_group)})")
            continue
    
    # Handle excluded length trajectories
    excluded_length_df = before_zscore_df[before_zscore_df["Length"] > 4].copy()
    if not excluded_length_df.empty:
        excluded_length_df["zIdPhi"] = 0
    
    # Combine all z-scored data
    zscored_df = pd.concat([zscored_valid_df, excluded_length_df], ignore_index=True)
    
    # Calculate threshold for VTE detection
    mean_zidphi = np.mean(zscored_df["zIdPhi"])
    std_zidphi = np.std(zscored_df["zIdPhi"])
    threshold = mean_zidphi + 1.5 * std_zidphi
    
    # Store all parameters for this rat
    training_params[rat] = {
        'choice_params': choice_params,
        'threshold': threshold
    }
    
    print(f"Training threshold for {rat}: {threshold}")

print(f"\nProcessed training data for {len(training_params)} rats")
print("Now processing testing data...")

# Second pass: Process testing data using training parameters
inference_path = paths.vte_values / "inferenceTesting"
if inference_path.exists():
    for rat_dir in inference_path.iterdir():
        rat = rat_dir.name
        if ".DS" in rat or ".csv" in rat:
            continue
        
        # Check if we have training parameters for this rat
        if rat not in training_params:
            logger.warning(f"No training parameters found for {rat}, skipping testing data")
            continue
        
        rat_testing_path = inference_path / rat
        testing_file = rat_testing_path / f"{rat}_Day1_trajectories.csv"
        
        if not testing_file.exists():
            logger.warning(f"Testing file not found: {testing_file}")
            continue
        
        # Load testing data
        testing_df = pd.read_csv(testing_file, header=0)
        
        if not all(col in testing_df.columns for col in ["IdPhi", "Choice", "ID", "Trial Type", "Length"]):
            logger.error(f"Missing required columns in testing data for {rat}")
            continue
        
        # Apply z-scoring using training parameters
        testing_df["zIdPhi"] = 0  # Default value
        
        choice_params = training_params[rat]['choice_params']
        threshold = training_params[rat]['threshold']
        
        # Apply z-scoring for each choice using training parameters
        for choice in testing_df["Choice"].unique():
            if choice in choice_params:
                choice_mask = testing_df["Choice"] == choice
                valid_length_mask = testing_df["Length"] <= 4
                combined_mask = choice_mask & valid_length_mask
                
                if combined_mask.any():
                    # Apply z-scoring using training mean and std
                    train_mean = choice_params[choice]['mean']
                    train_std = choice_params[choice]['std']
                    
                    if train_std > 0:  # Avoid division by zero
                        testing_df.loc[combined_mask, "zIdPhi"] = (
                            testing_df.loc[combined_mask, "IdPhi"] - train_mean
                        ) / train_std
                    else:
                        logger.warning(f"Zero std for choice {choice} in {rat}, setting zIdPhi to 0")
                        testing_df.loc[combined_mask, "zIdPhi"] = 0
            else:
                logger.warning(f"Choice {choice} not found in training data for {rat}")
        
        # Apply VTE classification using training threshold
        testing_df["VTE"] = testing_df["zIdPhi"] > threshold
        
        # Save results
        output_file = rat_testing_path / f"{rat}_Day1_zIdPhi.csv"
        testing_df.to_csv(output_file, index=False)
        
        print(f"Processed testing data for {rat}")
        print(f"  - Applied threshold: {threshold:.3f}")
        print(f"  - VTE instances: {sum(testing_df['VTE'])}/{len(testing_df)}")
        print(f"  - Saved to: {output_file}")

print("\nProcessing complete!")