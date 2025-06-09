import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Literal, Sequence

from config import settings

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
    
    if labels is None:
        labels = ["A", "B", "C", "D", "E", "F"]
    
    colors = ["green", "red", "blue", "orange", "purple", "pink"]
    for i, _ in enumerate(x_vals):
        plt.scatter(x_vals[i], y_vals[i], color=colors[i], alpha=0.4, label=labels[i])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if save:
        save_path = os.path.join(save, f"{settings.CURRENT_DAY}_population_scatter_plot.jpg")
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
    cdict: dict[Literal['red', 'green', 'blue', 'alpha'], Sequence[tuple[float, float, float]]] = {
        "red":   ((0.0,  0.0, 0.0),   # Black for zero
                  (0.01, 1.0, 1.0),   # Red for values just above zero
                  (1.0,  1.0, 1.0)),  # Keeping it red till the end

        "green": ((0.0,  0.0, 0.0),
                  (0.01, 0.0, 0.0),   # No green for low values
                  (1.0,  1.0, 1.0)),  # Full green at the end

        "blue":  ((0.0,  0.0, 0.0),
                  (0.01, 0.0, 0.0),   # No blue for low values
                  (1.0,  1.0, 1.0))}  # Full blue at the end

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