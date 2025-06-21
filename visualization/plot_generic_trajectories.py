import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

from config.paths import paths

vte_path = paths.vte_values / "BP13" / "Day8" / "trajectories.csv"

trajectories_csv = pd.read_csv(vte_path)
non_VTE_trajectory = None
VTE_trajectory = None
for index, row in trajectories_csv.iterrows():
    id = row["ID"]
    
    if id == "BP13_Day8_4":
        x = ast.literal_eval(row["X Values"])
        y = ast.literal_eval(row["Y Values"])
        
        non_VTE_trajectory = (x, y)
    
    if id == "BP13_Day8_14":
        x = ast.literal_eval(row["X Values"])
        y = ast.literal_eval(row["Y Values"])
        
        VTE_trajectory = (x, y)
        
dlc_path = paths.cleaned_dlc / "BP13" / "Day8_coordinates.csv"
coordinates_csv = pd.read_csv(dlc_path)
x_val = coordinates_csv["x"]
y_val = coordinates_csv["y"]

def plot_trajectory(x, y, trajectories_1, trajectories_2=None, title="", save=None, label_1=None, label_2=None, traj_id=None):
    """
    Plots up to two trajectories on the backdrop of x and y coordinates.

    Args:
        x (int array): x coordinates.
        y (int array): y coordinates.
        trajectories_1 (tuple): (trajectory_x_1, trajectory_y_1) where both are int arrays.
        trajectories_2 (tuple, optional): (trajectory_x_2, trajectory_y_2) if a second trajectory is desired. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "".
        save (str, optional): File path if saving is desired. Defaults to None.
        label_1 (Any, optional): Label for the first trajectory. Defaults to None.
        label_2 (Any, optional): Label for the second trajectory. Defaults to None.
        traj_id (str, optional): ID for saving the plot. Defaults to None.
    """
    
    # get trajectory points for the first trajectory
    trajectory_x_1, trajectory_y_1 = trajectories_1
    
    # plot the normal points (background)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color="green", alpha=0.4)
    
    # plot the first trajectory
    plt.plot(trajectory_x_1, trajectory_y_1, color="blue", alpha=0.8, label=label_1)
    
    # if the second trajectory exists, plot it
    if trajectories_2 is not None:
        trajectory_x_2, trajectory_y_2 = trajectories_2
        plt.plot(trajectory_x_2, trajectory_y_2, color="red", alpha=0.8, label=label_2)
    
    # add title and labels
    plt.title(title, fontsize=25)
    plt.xlabel("X coordinate", fontsize=18)
    plt.ylabel("Y coordinate", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=24)

    # saving logic
    if save is not None:
        if traj_id is not None:
            save_path = os.path.join(save, f"trajectory_{traj_id}.jpg")
        else:
            save_path = os.path.join(save, "trajectory.jpg")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


plot_trajectory(x_val, y_val, VTE_trajectory, non_VTE_trajectory, title="VTE vs Non-VTE Trajectories", label_1="VTE", label_2="Non-VTE")