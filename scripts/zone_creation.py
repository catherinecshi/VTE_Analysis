import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from scipy.spatial.qhull import ConvexHull
from mpl_point_clicker import clicker

from src import helper

# pylint: disable=import-error

def click_on_map(x_coords, y_coords):
    _, ax = plt.subplots()
    ax.scatter(x_coords, y_coords, color="red", marker=".", alpha=0.3)
    klicker = clicker(ax, ["vertex"])
    plt.show()
    
    return klicker.get_positions()["vertex"]

def make_hull(x_coords, y_coords, hull_points, reused=False):
    convex_hull = None

    def continue_plot(event=None):
        plt.close()
    
    def replot(event=None):
        nonlocal convex_hull
        convex_hull = None

        plt.close()
    
    _, ax = plt.subplots()
    ax.scatter(x_coords, y_coords, alpha=0.5, color="green", marker=".")
    
    if len(hull_points) < 3:
        print("Not enough points to construct hull")
        return None
    
    convex_hull = ConvexHull(hull_points)
    
    for simplex in convex_hull.simplices:
        ax.plot(hull_points[simplex, 0], hull_points[simplex, 1], "k-")
        
    hull_polygon = Polygon(hull_points[convex_hull.vertices], closed=True, edgecolor="k", fill=False)
    plt.gca().add_patch(hull_polygon)
    
    if reused:
        axes1 = plt.axes([0.8, 0.05, 0.1, 0.075])
        btn1 = Button(axes1, "Good")
        btn1.on_clicked(continue_plot)
        
        axes2 = plt.axes([0.2, 0.05, 0.1, 0.075])
        btn2 = Button(axes2, "Bad")
        btn2.on_clicked(replot)

    plt.show()
    return convex_hull

# get all the scatter plots of every rat for each day
dlc_path = os.path.join(helper.BASE_PATH, "processed_data", "dlc_data")
hull_path = os.path.join(helper.BASE_PATH, "processed_data", "hull_data")

for rat in os.listdir(dlc_path):
    if ".DS_Store" in rat:
        continue

    rat_path = os.path.join(dlc_path, rat)
    reused_hull = None
    for root, _, files in os.walk(rat_path):
        for f in files:
            if not "coordinates" in f:
                continue
            
            file_path = os.path.join(root, f)
            parts = f.split("_")
            day = parts[0]
            helper.update_day(day)
            
            df = pd.read_csv(file_path)
            x = df["x"]
            y = df["y"]
            
            if x.empty or y.empty:
                print(f"x or y is empty for {rat} on {day}")
                continue
            
            # reuse hull if it fits
            hull = None
            if reused_hull is not None:
                hull = make_hull(x, y, reused_hull, reused=True)
            
            current_hull_path = os.path.join(hull_path, f"{rat}_{day}_hull.npy")
            if hull is None:
                points = click_on_map(x, y)
                hull = make_hull(x, y, points)
                reused_hull = points[hull.vertices]
                np.save(current_hull_path, points[hull.vertices])
            else:
                np.save(current_hull_path, reused_hull)