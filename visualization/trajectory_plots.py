import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from utilities import math_utils

def plot_trajectory(x, y, trajectories, title="", save=None, label=None, traj_id=None):
    """
    plots a trajectory on the backdrop of x and y coordinates

    Args:
        x (int array): x coordinates
        y (int array): y coordinates
        trajectories (tuple): (trajectory_x, trajectory_y) where both are int arrays
        title (str, optional): title. Defaults to "".
        save (str, optional): file path if saving is desired. Defaults to None.
        label (Any, optional): label if desired. Defaults to None.
    """
    
    # get trajectory points
    trajectory_x, trajectory_y = trajectories
    
    # plot the normal points
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color="green", alpha=0.4)
    
    # plot the trajectory
    plt.plot(trajectory_x, trajectory_y, color="red", alpha=0.8, label=label)
    
    # add title
    plt.title(title)
    
    # display plot
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.legend()
    
    if save is not None and traj_id is not None:
        save_path = os.path.join(save, f"trajectory_{traj_id}.jpg")
        plt.savefig(save_path)
        plt.close()
    elif save is not None:
        save_path = os.path.join(save, "trajectory.jpg")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
def plot_trajectory_animation(x, y, trajectory_x=None, trajectory_y=None, interval=20, traj_id="", title="", label=None, save=None):
    """
    creates and displays an animation of a trajectory over the backdrop of x and y coordinates

    Args:
        x (int array): x coordinates
        y (int array): y coordinates
        trajectory_x (int array, optional): x coordinates of trajectory. Defaults to None.
        trajectory_y (int array, optional): y coordinates of trajectory. Defaults to None.
        interval (int, optional): time interval between frames in milliseconds. Defaults to 20.
        title (str, optional): title. Defaults to "".
        label (Any, optional): label. Defaults to None.
        save (str, optional): file path if saving is desired. Defaults to None.
    """
    
    if trajectory_x is None: # this is for when you want to plot the entire trajectory throughout the trial
        trajectory_x = x
    
    if trajectory_y is None:
        trajectory_y = y
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.2) # plot the totality first
    line, = ax.plot([], [], "bo-", linewidth = 2) # line plot
    
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    
    def init():
        # initialises animation with empty line plot
        line.set_data([], [])
        return line,
    
    def update(frame):
        # updates line plot for each frame
        # frame -> the current frame number
        
        x_val = trajectory_x[:frame] # get x up to the current frame
        y_val = trajectory_y[:frame]
        line.set_data(x_val, y_val)
        return line,

    # create animation
    ani = FuncAnimation(fig, update, frames = len(x), init_func = init, blit = True, interval = interval)
    
    # set title
    plt.title(title)
    
    if label:
        if isinstance(label, float):
            rounded_label = math_utils.round_to_sig_figs(label, 3)
        else:
            rounded_label = label
        legend_elements = [Line2D([0], [0], color="blue", lw=2, label=rounded_label)]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))
    
    # save or display
    if save:
        save_path = os.path.join(save, f"{traj_id}_trajectory_animation.gif")
        ani.save(save_path, writer="pillow")
        plt.close()
    else:
        plt.show()