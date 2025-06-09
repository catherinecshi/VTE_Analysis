"""
functions for working with 2D coordinates, regions, and shapes
"""

import numpy as np
from scipy.spatial.qhull import Delaunay

def is_point_in_ellipse(x, y, ellipse_params):
    """
    returns True if a 2D point is within an ellipse
    
    Parameters:
    - x (array)
    - y (array)
    - ellipse_params (dict):
        - should include 'center', 'width', 'height', and 'angle' as keys
    """
    
    center, width, height, angle = ellipse_params['center'], ellipse_params['width'], ellipse_params['height'], ellipse_params['angle']
    
    # Convert angle from degrees to radians for np.cos and np.sin
    theta = np.radians(angle)
    
    # Translate point to origin based on ellipse center
    x_translated = x - center[0]
    y_translated = y - center[1]
    
    # Rotate point by -theta to align with ellipse axes
    x_rotated = x_translated * np.cos(-theta) - y_translated * np.sin(-theta)
    y_rotated = x_translated * np.sin(-theta) + y_translated * np.cos(-theta)
    
    # Check if rotated point is inside the ellipse
    if (x_rotated**2 / (width/2)**2) + (y_rotated**2 / (height/2)**2) <= 1:
        return True  # Inside the ellipse
    else:
        return False  # Outside the ellipse

def is_point_in_hull(point, hull):
    """returns True if a 2D point is within the convex hull"""
    
    del_tri = Delaunay(hull.points[hull.vertices])
    
    # check if point is inside the hull
    return del_tri.find_simplex(point, bruteforce=False, tol=None) >= 0