import pandas as pd
import numpy as np
import ast

from config.paths import paths

def calculate_trajectory_length(x_values, y_values):
    """
    Calculate the spatial length of a trajectory given X and Y coordinates.
    
    Args:
        x_values (list): List of X coordinates
        y_values (list): List of Y coordinates
    
    Returns:
        float: Total spatial length of the trajectory
    """
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(x_values)):
        # Calculate Euclidean distance between consecutive points
        dx = x_values[i] - x_values[i-1]
        dy = y_values[i] - y_values[i-1]
        distance = np.sqrt(dx**2 + dy**2)
        total_length += distance
    
    return total_length

def parse_coordinate_string(coord_string):
    """
    Parse a string representation of a list into an actual list.
    
    Args:
        coord_string (str): String representation of coordinates
    
    Returns:
        list: Parsed coordinates
    """
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        return ast.literal_eval(coord_string)
    except (ValueError, SyntaxError):
        print(f"Error parsing coordinates: {coord_string}")
        return []

def process_trajectory_file(trajectory_file_path, zidphi_file_path):
    """
    Process a single trajectory file and its corresponding zIdPhi file.
    
    Args:
        trajectory_file_path (Path): Path to the trajectory CSV file
        zidphi_file_path (Path): Path to the zIdPhi CSV file
    
    Returns:
        pd.DataFrame: Processed data with additional columns
    """
    # Read trajectory data
    try:
        trajectory_df = pd.read_csv(trajectory_file_path)
    except Exception as e:
        print(f"Error reading trajectory file {trajectory_file_path}: {e}")
        return pd.DataFrame()
    
    # Read zIdPhi data
    try:
        zidphi_df = pd.read_csv(zidphi_file_path)
    except Exception as e:
        print(f"Error reading zIdPhi file {zidphi_file_path}: {e}")
        return pd.DataFrame()
    
    # Calculate spatial trajectory length and average speed
    spatial_lengths = []
    average_speeds = []
    
    for _, row in trajectory_df.iterrows():
        # Parse X and Y coordinates
        x_coords = parse_coordinate_string(row['X Values'])
        y_coords = parse_coordinate_string(row['Y Values'])
        
        # Calculate spatial length
        spatial_length = calculate_trajectory_length(x_coords, y_coords)
        spatial_lengths.append(spatial_length)
        
        # Calculate average speed (spatial length / time length)
        time_length = row['Length']
        if time_length > 0:
            avg_speed = spatial_length / time_length
        else:
            avg_speed = 0.0
        average_speeds.append(avg_speed)
    
    # Add new columns to trajectory dataframe
    trajectory_df['Spatial_Length'] = spatial_lengths
    trajectory_df['Average_Speed'] = average_speeds
    
    # Add this before the merge in process_trajectory_file()
    duplicate_ids = zidphi_df[zidphi_df.duplicated(subset=['ID'], keep=False)]
    if not duplicate_ids.empty:
        print(f"Warning: Duplicate IDs found in zIdPhi data: {duplicate_ids['ID'].unique()}")
    
    # Merge with zIdPhi data based on ID
    merged_df = trajectory_df.merge(zidphi_df[['ID', 'zIdPhi', 'VTE']], 
                                   on='ID', 
                                   how='left')
    
    return merged_df

def process_all_trajectories(base_path, centralized_data_path):
    """
    Process all trajectory files in the directory structure.
    
    Args:
        base_path (Path): Base directory containing rat folders
    """
    # Create centralized_data directory if it doesn't exist
    centralized_data_path.mkdir(exist_ok=True)
    
    # Find all trajectory files
    trajectory_files = []
    for rat_folder in base_path.iterdir():
        if rat_folder.is_dir() and rat_folder.name != 'centralized_data':
            for day_folder in rat_folder.iterdir():
                if day_folder.is_dir():
                    # Look for trajectory file
                    trajectory_file = day_folder / f"{rat_folder.name}_{day_folder.name}_trajectories.csv"
                    if trajectory_file.exists():
                        # Look for corresponding zIdPhi file
                        zidphi_file = day_folder / f"zIdPhi_day_{day_folder.name}.csv"
                        if zidphi_file.exists():
                            trajectory_files.append((trajectory_file, zidphi_file))
                        else:
                            print(f"Warning: No zIdPhi file found for {trajectory_file}")
    
    print(f"Found {len(trajectory_files)} trajectory files to process")
    
    # Process each trajectory file
    for trajectory_file, zidphi_file in trajectory_files:
        print(f"Processing {trajectory_file.name}...")
        
        # Process the files
        try:
            processed_df = process_trajectory_file(trajectory_file, zidphi_file)
        except Exception as e:
            print(f"ERROR {e} FOR RAT {rat_folder} ON {day_folder}")
        
        if not processed_df.empty:
            # Save to centralized_data folder
            output_filename = f"processed_{trajectory_file.name}"
            output_path = centralized_data_path / output_filename
            processed_df.to_csv(output_path, index=False)
            print(f"Saved processed data to {output_path}")
        else:
            print(f"No data to save for {trajectory_file.name}")

def main():
    """
    Main function to run the trajectory analysis.
    """
    base_path = paths.vte_values
    print(f"Processing trajectories from: {base_path}")
    process_all_trajectories(base_path, paths.central)
    print("Processing complete!")

if __name__ == "__main__":
    main()