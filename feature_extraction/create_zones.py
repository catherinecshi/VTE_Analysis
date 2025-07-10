"""
Main script for creating zones to find VTEs in.

Define polygons (convex hulls) by the user clicking on points. Those polygons are used further
in the pipeline to define the starts and ends of trajectories. The polygons are automatically
assumed to be reusable across different days for the same rat. The user will be able to
specify if the same polygons fit across different days.

To use, run the code after passing in your own folder paths you want to save everything to.
    - dlc_path -> folder with all the dlc files.
        - assumes dlc_path > rat > day.csv
        - assumes there is a "x" and "y" column
    - hull_path -> folder you want to save the zone polygons into
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from scipy.spatial.qhull import ConvexHull
from mpl_point_clicker import clicker

from config import settings
from config.paths import paths

# pylint: disable=import-error

def click_on_map(x_coords, y_coords):
    """
    Creates the plot and the clicker that takes clicks as inputs.
    
    Args:
        x_coords (float array): X coords
        y_coords (float array): Y coords
    
    Returns:
        Array of (x, y) coordinates of where the user clicked
    """
    
    _, ax = plt.subplots()
    ax.scatter(x_coords, y_coords, color="red", marker=".", alpha=0.3)
    klicker = clicker(ax, ["vertex"])
    plt.show()
    
    return klicker.get_positions()["vertex"]

def make_hull(x_coords, y_coords, hull_points, reused=False):
    """
    creates a polygon (convex hull) based on the points from click_on_map
    it then presents a plot of the convex hull to verify with the user
    if the coordinates used is a different day than the one the polygon was
    created in, then the user is prompted about whether the polygon actually
    fits onto the coordinates.
    
    Args:
        x_coords (float array): X coordinates
        y_coords (float array): Y coordinates
        hull_points (array): Points to construct hull from
        reused (bool): Whether it is the same day as the one hull was created from
    """
    
    convex_hull = None

    def continue_plot(event=None):
        """Callback for accepting the hull"""
        plt.close()
    
    def replot(event=None):
        """Callback for rejecting the hull"""
        nonlocal convex_hull
        convex_hull = None

        plt.close()
    
    _, ax = plt.subplots()
    ax.scatter(x_coords, y_coords, alpha=0.5, color="green", marker=".")
    
    if len(hull_points) < 3:
        print("Not enough points to construct hull")
        return None
    
    # create and plot the polygon
    convex_hull = ConvexHull(hull_points)
    for simplex in convex_hull.simplices:
        ax.plot(hull_points[simplex, 0], hull_points[simplex, 1], "k-")
        
    hull_polygon = Polygon(hull_points[convex_hull.vertices], closed=True, edgecolor="k", fill=False)
    plt.gca().add_patch(hull_polygon)
    
    # display buttons if this is a different day that the one used to create polygon
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
dlc_path = paths.cleaned_dlc / "inferenceTesting"
hull_path = paths.hull_data / "inferenceTesting"

#for rat in os.listdir(dlc_path):
    #if ".DS_Store" in rat:
       # continue

    #rat_path = os.path.join(dlc_path, rat)
reused_hull = None # reset
for root, _, files in os.walk(dlc_path):
    for f in files:
        if not "coordinates" in f:
            continue
        
        file_path = os.path.join(root, f)
        parts = f.split("_")
        rat = parts[0]
        #day = parts[0]
        #settings.update_day(day)
        
        df = pd.read_csv(file_path)
        x = df["x"]
        y = df["y"]
        
        if x.empty or y.empty:
            #print(f"x or y is empty for {rat} on {day}")
            print(f"x or y is empty for {f}")
            continue
        
        # reuse hull if it fits
        hull = None
        if reused_hull is not None:
            hull = make_hull(x, y, reused_hull, reused=True)
        
        #current_hull_path = os.path.join(hull_path, f"{rat}_{day}_hull.npy")
        current_hull_path = os.path.join(hull_path, f"{rat}_hull_test.npy")
        if hull is None:
            points = click_on_map(x, y)
            hull = make_hull(x, y, points)
            reused_hull = points[hull.vertices]
            np.save(current_hull_path, points[hull.vertices])
        else:
            np.save(current_hull_path, reused_hull)