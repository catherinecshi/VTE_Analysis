import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

def create_scatter_plot(x, y, title = '', xlabel = '', ylabel = '', save = None):
    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, color='green', alpha=0.4)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if save:
        plt.savefig(save)
    else:
        plt.show()
    
def create_occupancy_map(x, y, framerate = 0.03, bin_size = 15, title = 'Occupancy Map', save = None):
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
    cdict = {'red':   [(0.0,  0.0, 0.0),   # Black for zero
                       (0.01, 1.0, 1.0),   # Red for values just above zero
                       (1.0,  1.0, 1.0)],  # Keeping it red till the end

             'green': [(0.0,  0.0, 0.0),
                       (0.01, 0.0, 0.0),   # No green for low values
                       (1.0,  1.0, 1.0)],  # Full green at the end

             'blue':  [(0.0,  0.0, 0.0),
                       (0.01, 0.0, 0.0),   # No blue for low values
                       (1.0,  1.0, 1.0)]}  # Full blue at the end

    custom_cmap = LinearSegmentedColormap('custom_hot', segmentdata=cdict, N=256)
    
    # rotating so it looks like scatter plot
    rotated_occupancy_grid = np.rot90(occupancy_grid)
    
    # plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(rotated_occupancy_grid, cmap=custom_cmap, interpolation='nearest')
    plt.colorbar(label='Time spent in seconds')
    plt.title(title)
    plt.xlabel('X Bins')
    plt.ylabel('Y Bins')
    
    if save:
        plt.savefig(save)
    else:
        plt.show()
        
def plot_hull(x, y, hull_points, densest_cluster_points, hull, save = None):
    plt.scatter(x, y)

    # Plotting (optional, for visualization)
    plt.scatter(hull_points[:,0], hull_points[:,1], alpha=0.5, color = 'green')
    plt.scatter(densest_cluster_points[:,0], densest_cluster_points[:,1], color='red')
    for simplex in hull.simplices:
        plt.plot(densest_cluster_points[simplex, 0], densest_cluster_points[simplex, 1], 'k-')

    # Create a Polygon patch for the convex hull
    hull_polygon = Polygon(densest_cluster_points[hull.vertices], closed=True, edgecolor='k', fill=False)
    plt.gca().add_patch(hull_polygon)
    
    if save:
        plt.savefig(save)
    else:
        plt.show()
        
def plot_ellipse(ellipse_params, x, y):
    fig, ax = plt.subplots()
    
    # Plot data points
    ax.scatter(x, y, alpha=0.5)
    
    # If ellipse_params is not None, create and add the ellipse
    if ellipse_params is not None:
        ellipse = Ellipse(xy=ellipse_params['center'], width=ellipse_params['width'],
                          height=ellipse_params['height'], angle=ellipse_params['angle'],
                          edgecolor='r', facecolor='none')
        ax.add_patch(ellipse)
    
    plt.show()