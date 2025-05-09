"""
General plotting functions
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

from src import helper

#matplotlib setup
#import matplotlib
#matplotlib.use("Agg")

### GENERIC PLOTS ------------
def create_scatter_plot(x, y, title="", xlabel="", ylabel="", save=None):
    """
    create scatter plot

    Args:
        x (int array): x coordinates
        y (int array): y coordinates
        title (str, optional): title. Defaults to ''.
        xlabel (str, optional): x label. Defaults to ''.
        ylabel (str, optional): y label. Defaults to ''.
        save (str, optional): file path if saving is desired. Defaults to None.
    """
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="green", alpha=0.4)
    
    plt.title(title, fontsize=25)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(False)
    
    if save:
        save_path = f"{save}/scatter_plot.jpg"
        plt.savefig(save_path)
    else:
        plt.show()

def create_populational_scatter_plot(x_vals, y_vals, labels=None, title="", xlabel="", ylabel="", save=None):
    plt.figure(figsize=(10, 6))
    
    colors = ["green", "red", "blue", "orange", "purple", "pink"]
    for i, _ in enumerate(x_vals):
        plt.scatter(x_vals[i], y_vals[i], color=colors[i], alpha=0.4, label=labels[i])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if save:
        save_path = os.path.join(save, f"{helper.CURRENT_DAY}_population_scatter_plot.jpg")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_bar_plot(data, x_ticks, errors=None, xlim=None, ylim=None, significance_pairs=None, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    
    # Convert proportions to percentages
    data_percentages = [d * 100 for d in data]
    if errors:
        errors_percentages = [e * 100 for e in errors]
    
    if errors:
        bars = plt.bar(x_ticks, data_percentages, yerr=errors_percentages, capsize=5)
    else:
        bars = plt.bar(x_ticks, data_percentages)
    
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    if xlim:
        plt.xlim(xlim)
    
    if ylim:
        # Convert proportion limits to percentage limits
        plt.ylim((ylim[0] * 100, ylim[1] * 100))
    
    plt.ylim(0, 10)
    
    if significance_pairs:
        bar_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]  # Get center positions of bars
        bar_heights = [bar.get_height() for bar in bars]  # Get heights of the bars
        
        max_height = max(bar_heights)  # To adjust the height of the significance lines
        
        # Significance between first and second bars
        if significance_pairs[0]:
            plt.plot([bar_positions[0], bar_positions[1]], [max_height * 1.05, max_height * 1.05], color='black')
            plt.text((bar_positions[0] + bar_positions[1]) / 2, max_height * 1.07, '*', ha='center', va='bottom', fontsize=24)
        
        # Significance between second and third bars
        if significance_pairs[1]:
            plt.plot([bar_positions[1], bar_positions[2]], [max_height * 1.15, max_height * 1.15], color='black')
            plt.text((bar_positions[1] + bar_positions[2]) / 2, max_height * 1.17, '*', ha='center', va='bottom', fontsize=24)
        
        # Significance between first and third bars
        if significance_pairs[2]:
            plt.plot([bar_positions[0], bar_positions[2]], [max_height * 1.25, max_height * 1.25], color='black')
            plt.text((bar_positions[0] + bar_positions[2]) / 2, max_height * 1.27, '*', ha='center', va='bottom', fontsize=24)
    
    plt.tight_layout()
    plt.show()

def create_line_plot(x, y, sem, xlim=None, ylim=None, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    
     # split into two lines if any label or title is too long
    max_title_length = 40
    if len(title) > max_title_length:
        title = "\n".join([title[:max_title_length], title[max_title_length:]])
    
    max_label_length = 50
    if len(xlabel) > max_label_length:
        xlabel = "\n".join([xlabel[:max_label_length], xlabel[max_label_length:]])
    
    if len(ylabel) > max_label_length:
        ylabel = "\n".join([ylabel[:max_label_length], ylabel[max_label_length:]])
    
    plt.errorbar(x, y, yerr=sem, fmt="-o", capsize=5, linewidth=5)
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    if xlim:
        plt.xlim(xlim)
    
    if ylim:
        plt.ylim(ylim)
    
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_cumulative_frequency(data, title="", xlabel="", ylabel=""):
    sorted_data = np.sort(data)
    
    # x label
    cum_freq = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, cum_freq, marker=".", linestyle="-", color="b")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.grid(True)
    plt.show()

def create_histogram(df, x, y, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=x, hue=y, multiple="stack", kde=True, legend=False, binwidth=1)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()

def create_frequency_histogram(list1, label1="", list2=None, label2="", binwidth=None, stat="density", xlim=None, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=list1, kde=True, color="blue", label=label1, binwidth=binwidth, stat=stat)
    
    # Plot the second dataset if provided
    if list2 is not None:
        sns.histplot(data=list2, kde=True, color="red", label=label2, binwidth=binwidth, stat=stat)
    
    plt.legend(fontsize=24)
    
    if xlim is not None:
        plt.xlim(xlim)
        
    # split into two lines if any label or title is too long
    max_title_length = 40
    if len(title) > max_title_length:
        title = "\n".join([title[:max_title_length], title[max_title_length:]])
    
    max_label_length = 30
    if len(xlabel) > max_label_length:
        xlabel = "\n".join([xlabel[:max_label_length], xlabel[max_label_length:]])
    
    if len(ylabel) > max_label_length:
        ylabel = "\n".join([ylabel[:max_label_length], ylabel[max_label_length:]])
    
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

def create_multiple_frequency_histograms(dictionary, binwidth=None, stat="density", xlim=None, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    
    colors = ["red", "blue", "green", "orange", "purple", "pink"]
    for i, (key, value) in enumerate(dictionary.items()):
        sns.histplot(data=value, kde=True, color=colors[i], label=key, binwidth=binwidth, stat=stat)
    
    plt.legend()
    
    if xlim is not None:
        plt.xlim(xlim)
    
    plt.title(title, fontsize=25)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.show()
    

def create_box_and_whisker_plot(df, x, y, xlim=None, title="", xlabel="", ylabel=""):
    filtered_df = df.groupby(x).filter(lambda group: len(group[y]) >= 5)
    
    if xlim:
        unique_x_values = filtered_df[x].unique()[:xlim]
        filtered_df = filtered_df[filtered_df[x].isin(unique_x_values)]
    
    plt.figure(figsize=(10, 6))
    filtered_df.boxplot(column=y, by=x, grid=False)
    
    plt.suptitle("")
    plt.title(title, fontsize=25)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.show()

def create_occupancy_map(x, y, framerate=0.03, bin_size=15, title="", xlabel="", ylabel="", save=None):
    """
    creates occupancy map

    Args:
        x (int array): x coordinates
        y (int array): y coordinates
        framerate (float, optional): number of seconds between each x/y coordinate. Defaults to 0.03.
        bin_size (int, optional): how big/small each bin on map should be. Defaults to 15.
        title (str, optional): title. Defaults to ''.
        save (str, optional): file path if saving is desired. Defaults to None.
    """
    
    # determine size of occupancy map
    x_max = x.max()
    y_max = y.max()
    num_bins_x = int(np.ceil(x_max / bin_size))
    num_bins_y = int(np.ceil(y_max / bin_size))
    
    # empty grid
    occupancy_grid = np.zeros((num_bins_x, num_bins_y))
    
    # bin data points
    for i in range(len(x)):
        bin_x = int(x.iloc[i] // bin_size)
        bin_y = int(y.iloc[i] // bin_size)
        
        occupancy_grid[bin_x, bin_y] += 1
    
    occupancy_grid = occupancy_grid / framerate
    
    # Define the colors for the custom colormap
    cdict = {"red":   [(0.0,  0.0, 0.0),   # Black for zero
                       (0.01, 1.0, 1.0),   # Red for values just above zero
                       (1.0,  1.0, 1.0)],  # Keeping it red till the end

             "green": [(0.0,  0.0, 0.0),
                       (0.01, 0.0, 0.0),   # No green for low values
                       (1.0,  1.0, 1.0)],  # Full green at the end

             "blue":  [(0.0,  0.0, 0.0),
                       (0.01, 0.0, 0.0),   # No blue for low values
                       (1.0,  1.0, 1.0)]}  # Full blue at the end

    custom_cmap = LinearSegmentedColormap("custom_hot", segmentdata=cdict, N=256)
    
    # rotating so it looks like scatter plot
    rotated_occupancy_grid = np.rot90(occupancy_grid)
    
    # plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(rotated_occupancy_grid, cmap=custom_cmap, interpolation="nearest")
    plt.colorbar(label="Time spent in seconds")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save:
        save_path = f"{save}/occupancy_map.jpg"
        plt.savefig(save_path)
    else:
        plt.show()



### PLOTTING ZONES ------------
def plot_convex_hull(x, y, hull_points, title="", xlabel="", ylabel=""):
    hull = ConvexHull(hull_points)
    
    plt.plot(x, y, "o", markersize=5)
    for simplex in hull.simplices:
        plt.plot(hull_points[simplex, 0], hull_points[simplex, 1], "k-")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.show()

def plot_hull_from_density(x, y, hull_points, densest_cluster_points, hull, save=None, title=""):
    """
    plots a convex hull on a backdrop of x and y coordinates

    Args:
        x (int array): x coordinates
        y (int array): y coordinates
        hull_points (int/float array): (x_points, y_points) coordinates of points forming the convex hull
        densest_cluster_points (int/float array): (x_points, y_points) coordinates of points in densest cluster
        hull (scipy.spatial.ConvexHull): contains simplices and vertices of hull
        save (str, optional): file path if saving is desired. Defaults to None.
        title (str, optional): title. Defaults to "".
    """
    
    plt.scatter(x, y)

    # Plotting (optional, for visualization)
    plt.scatter(hull_points[:,0], hull_points[:,1], alpha=0.5, color="green")
    plt.scatter(densest_cluster_points[:,0], densest_cluster_points[:,1], color="red")
    for simplex in hull.simplices:
        plt.plot(densest_cluster_points[simplex, 0], densest_cluster_points[simplex, 1], "k-")

    # Create a Polygon patch for the convex hull
    hull_polygon = Polygon(densest_cluster_points[hull.vertices], closed=True, edgecolor="k", fill=False)
    plt.gca().add_patch(hull_polygon)
    
    plt.title(title) # add title
    
    if save:
        save_path = f"{save}/convex_hull.jpg"
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_hull_from_indices(x, y, hull_indices):
    """
    Plot the convex hull for given x and y coordinates and precomputed hull indices.

    Parameters:
    x (array-like): An array of x coordinates
    y (array-like): An array of y coordinates
    hull_indices (array-like): Indices of the points that make up the convex hull
    """
    points = np.column_stack((x, y))
    plt.plot(points[:, 0], points[:, 1], 'o')
    
    hull_points = points[hull_indices]
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2)
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'ro')
    
    # Close the hull by connecting the last point to the first
    plt.plot([hull_points[-1, 0], hull_points[0, 0]], [hull_points[-1, 1], hull_points[0, 1]], 'r--', lw=2)
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Convex Hull')
    plt.show()
        
def plot_ellipse(ellipse_params, x, y, save=None, title=""):
    """
    plots ellipse on a backdrop of x & y coords

    Args:
        ellipse_params (dict): has the following keys
            - 'center': floats (x, y) corresponding to center
            - 'width': float representing width
            - 'height': float representing height
            - 'angle': float representing the rotational angle of ellipse in degrees
        x (int array): x coordinates
        y (int array): y coordinates
        save (str, optional): file path if saving is desired. Defaults to None.
        title (str, optional): title. Defaults to "".
    """
    
    _, ax = plt.subplots()
    
    # Plot data points
    ax.scatter(x, y, alpha=0.5)
    
    # If ellipse_params is not None, create and add the ellipse
    if ellipse_params is not None:
        ellipse = Ellipse(xy=ellipse_params["center"], width=ellipse_params["width"],
                          height=ellipse_params["height"], angle=ellipse_params["angle"],
                          edgecolor="r", facecolor="none")
        ax.add_patch(ellipse)
    
    plt.title(title) # add title
    
    if save:
        save_path = f"{save}/ellipse.jpg"
        plt.savefig(save_path)
    else:
        plt.show()



### PLOTTING TRAJECTORIES --------
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
            rounded_label = helper.round_to_sig_figs(label, 3)
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


### SPECIALIZED PLOTS -------
def plot_zIdPhi(zIdPhi_values, save=None):
    """
    plot the mean and std of zIdPhi values

    Args:
        zIdPhi_values (dict):{choice: zIdPhis}
        save (str, optional): file path if saving is desired. Defaults to None.
    """
    
    # Collect all zIdPhi values from all trial types
    all_zIdPhis = []
    for zIdPhis in zIdPhi_values.values():
        all_zIdPhis.extend(zIdPhis)
    
    # Convert to a NumPy array for statistical calculations
    all_zIdPhis = np.array(all_zIdPhis)
    
    # Create a single plot
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(all_zIdPhis, bins=30, alpha=0.7, label="All Trial Types")
    
    # Calculate and plot the mean and standard deviation lines
    mean = np.mean(all_zIdPhis)
    std = np.std(all_zIdPhis)
    
    plt.axvline(mean, color="red", linestyle="dashed", linewidth=2, label="Mean")
    plt.axvline(mean + std, color="green", linestyle="dashed", linewidth=2, label="+1 STD")
    plt.axvline(mean - std, color="green", linestyle="dashed", linewidth=2, label="-1 STD")
    
    # Set the title and labels
    plt.title("Combined IdPhi Distribution Across All Trial Types")
    plt.xlabel("zIdPhi")
    plt.ylabel("Frequency")
    
    # Show the legend
    plt.legend()
    plt.tight_layout()
    
    if save:
        save_path = f"{save}/zIdPhi_Distribution.jpg"
        plt.savefig(save_path)
    else:
        plt.show()