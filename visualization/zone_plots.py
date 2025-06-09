import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from scipy.spatial.qhull import ConvexHull

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