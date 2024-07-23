import os
import pandas as pd

from src import creating_zones

base_path = "/Users/catpillow/Documents/VTE_Analysis"
dlc_path = os.path.join(base_path, "processed_data", "dlc_data")

for rat in os.listdir(dlc_path):
    rat_path = os.path.join(dlc_path, rat)
    
    for root, _, files in os.walk(rat_path):
        for f in files:
            if not "coordinates" in f:
                continue
            
            file_path = os.path.join(root, f)
            parts = f.split("_")
            
            df = pd.read_csv(file_path)
            x = df["x"]
            y = df["y"]
            
            lines = creating_zones.generate_lines(x, y, plot=True)
            