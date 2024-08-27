"""
Experimental method for how to define zones on a radial maze when the camera isn't consistent across sessions
Currently Includes
    - get_centre_zone

The basic idea/presuppositions:
    - After filtering, the points in the dataframe for the rat's coordinates should correspond to where the rat has been
    - When plotted out, the paths the rat can take probably correspond somewhat to straight lines
        - such that on a radial arm maze, there are 4 straight lines/paths that all intersect somewhat in the middle
    - If you define the clusters at the ends of the lines & at the intersection, you should have the centre & arm zones

Procedure:

    1. Generate lines that cover most of the plot
        - does so by incrementing the x or y intercept for each slope

    2. Check how well each line is covered by points from the dataframe
        - divide each line into segments
        - check if there are any points within the distance threshold to the segment (maybe check for more points than one?)
        - count the number of consecutive segments that have coverage
        - if over a threshold, consider the line to correspond to covering a path that the rat has taken
        
    3. Make a new set of lines that only include that have enough coverage
    
    4. Check for the part that has the densest concentration of intersections
        - presumably this will be in the start
        
    5. Create a convex hull that encapsulates the cluster surrounding that zone

"""

import os
import logging
import alphashape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.qhull import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib.patches import Polygon as mPolygon
from shapely.geometry import Polygon as sPolygon
from shapely.geometry import Point
from datetime import datetime

from src import helper

### LOGGING
logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# makes a new log everytime the code runs by checking the time
log_file = datetime.now().strftime("/Users/catpillow/Documents/VTE_Analysis/doc/creating_zones_log_%Y%m%d_%H%M%S.txt")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

### PYLINT
# pylint: disable=no-name-in-module, consider-using-enumerate


def calculate_range(x, y):
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    
    return x_min, x_max, y_min, y_max

def create_lines(x_coords, y_coords, save=None):
    """
    Generate a random set of lines within bounds (the x and y coordinates 
    that bound the points in dataframe) that cover most of the bounds

    Args:
        x (array): x coords from df
        y (array): y coords from df
        plot (bool, optional): whether to plot the lines, 
                               mostly to check if they've been generated in a way to covers the bounds well

    Returns:
        list: list of tuples representing the line. includes the slope (m) & y-intercept (b) for angled lines
    """
    x_min, x_max, y_min, y_max = calculate_range(x_coords, y_coords)
    num_lines = 3000
    x_positions = np.random.uniform(x_min, x_max, num_lines)
    y_positions = np.random.uniform(y_min, y_max, num_lines)
    angles = np.random.uniform(0, 180, num_lines)
    
    lines = []
    for x, y, angle in zip(x_positions, y_positions, angles):
        theta = np.radians(angle)
        if np.isclose(np.cos(theta), 0): # vertical
            lines.append((float("inf"), x))
        elif np.isclose(np.sin(theta), 0): # horizontal
            lines.append((0, y))
        else:
            m = np.tan(theta)
            b = y - m * x
            lines.append((m, b))
    
    # explicitly make horizontal and vertical lines
    for i, x in enumerate(x_positions):
        if i > 250:
            break
        lines.append((float("inf"), x))
    
    for i, y in enumerate(y_positions):
        if i > 250:
            break
        lines.append((0, y))
    if save:
        plot_lines(x_coords, y_coords, lines, save)
    
    return lines

def calculate_line_coverages(x, y, lines, num_segments=15, threshold=10, save=None):
    """
    calculates how well each line is covered by points based on how many line segments have points in its vicinity

    Args:
        x (np array): x values
        y (np array): y values
        lines (list): list including slope and intercept values, making up y = mx + b
        num_segments (int, optional): the number of segments for testing lines' fit. Defaults to 15.
        threshold (int, optional): distance a point can be from a line before it's not considered in its vicinity. Defaults to 5.
        plot (bool, optional): whether to plot the mean and std of coverage scores to know what a good threshold is. Defaults to False.

    Returns:
        coverage_scores (int array): how many segments on the line are covered by points
        starts (int array): the start of where the line is being covered
        ends (int array): the end of where the line is being covered
        
    Procedure:
        1. for vertical lines
            - calculate segment_length, and use that to determine the start and stop of each segment
            - check how far each point within the y range of the segment is from the x coord of the line
            - if there is a point that falls within the distance threshold for how close it is to the line,
                add the start and end of the segment to current_streak_segments
                and update longest_streak if current_streak_segments is larger than longest_streak
            - if there isn't a point, then reset current_streak_segments, but not longest_streak
            - longest_streak is added as the coverage score
        2. same for horizontal lines, but swap x and y
        3. same again for angled lines, but just with different ways of divying up segments because the length isn't the same for every line
    """
    
    coverage_scores = []
    starts = [] # start of the cut out line - based upon longest_streak
    ends = [] # end of the cut out line
    points = np.array(list(zip(x, y)))
    x_min, x_max, y_min, y_max = calculate_range(x, y)
    
    for slope, b in lines:
        consecutive_coverage = 0 # what i'm looking for is consecutive coverage, much like a line
        longest_streak = 0
        current_streak_segments = []
        start = None
        end = None
        
        # vertical lines are diff, so i'm just seeing how many points are near the line
        if np.isinf(slope) or np.abs(slope) > 1e10:
            segment_length = (y_max - y_min) / num_segments

            for i in range(num_segments):
                segment_start = y_min + i * segment_length
                segment_end = segment_start + segment_length
                segment_points = points[(y >= segment_start) & (y <= segment_end)] # points within segment
                
                if len(segment_points) > 0: # check if there are any points
                    distances = np.abs(segment_points[:, 0] - b)
                    point_on_line = np.any(distances <= threshold)
                    
                    # add to consecutive coverage if there is something on line, no if not
                    if point_on_line:
                        current_streak_segments.append((segment_start, segment_end))
                        consecutive_coverage += 1
                        if longest_streak < consecutive_coverage:
                            longest_streak = consecutive_coverage # keep track of longest streak of coverage
                    else:
                        if longest_streak == consecutive_coverage and consecutive_coverage > 0:
                            start = current_streak_segments[0][0]
                            end = current_streak_segments[-1][-1]
                        current_streak_segments.clear()
                        consecutive_coverage = 0
            
            if longest_streak == consecutive_coverage and consecutive_coverage > 0:
                start = current_streak_segments[0][0]
                end = current_streak_segments[-1][-1]
            
            if longest_streak > 0:
                starts.append(start)
                ends.append(end)
            else:
                starts.append(0)
                ends.append(0)
            
            coverage = longest_streak / num_segments
            coverage_scores.append(coverage)
            
            continue
        
        # handle horizontal lines much like vertical lines
        if np.abs(slope) < 1e-10:
            segment_length = (x_max - x_min) / num_segments

            for i in range(num_segments):
                segment_start = x_min + i * segment_length
                segment_end = segment_start + segment_length
                segment_points = points[(x >= segment_start) & (x <= segment_end)]
                
                if len(segment_points) > 0: # check if there are any points
                    distances = np.abs(segment_points[:, 1] - b)
                    point_on_line = np.any(distances <= threshold)
                    
                    # add to consecutive coverage if there is something on line, no if not
                    if point_on_line:
                        current_streak_segments.append((segment_start, segment_end))
                        consecutive_coverage += 1
                        if longest_streak < consecutive_coverage:
                            longest_streak = consecutive_coverage # keep track of longest streak of coverage
                    else:
                        if longest_streak == consecutive_coverage and consecutive_coverage > 0:
                            start = current_streak_segments[0][0]
                            end = current_streak_segments[-1][-1]
                        current_streak_segments.clear()
                        consecutive_coverage = 0
            
            if longest_streak == consecutive_coverage and consecutive_coverage > 0:
                start = current_streak_segments[0][0]
                end = current_streak_segments[-1][-1]
            
            if longest_streak > 0:
                starts.append(start)
                ends.append(end)
            else:
                starts.append(0)
                ends.append(0)
            
            coverage = longest_streak / num_segments
            coverage_scores.append(coverage)
            
            continue
        
        # non vertical or horizontal lines
        # find start and end of x values to divide into segments
        filtered_min = x[x > x_min]
        if not filtered_min.empty:
            line_x_min = filtered_min.min()
        
        filtered_max = x[x < x_max]
        if not filtered_max.empty:
            line_x_max = filtered_max.max()
            
        segment_length = (line_x_max - line_x_min) / num_segments

        for i in range(num_segments):
            segment_start = x_min + i * segment_length
            segment_end = segment_start + segment_length
            segment_points = points[(x >= segment_start) & (x <= segment_end)]
            
            if len(segment_points) > 0:
                distances = np.abs(slope * segment_points[:, 0] - segment_points[:, 1] + b) / np.sqrt(slope ** 2 + 1)
                point_on_line = np.any(distances <= threshold)
                    
                # add to consecutive coverage if there is something on line, no if not
                if point_on_line:
                    current_streak_segments.append((segment_start, segment_end))
                    consecutive_coverage += 1
                    if longest_streak < consecutive_coverage:
                        longest_streak = consecutive_coverage # keep track of longest streak of coverage
                else:
                    if longest_streak == consecutive_coverage and consecutive_coverage > 0:
                        start = current_streak_segments[0][0]
                        end = current_streak_segments[-1][-1]
                    current_streak_segments.clear()
                    consecutive_coverage = 0
        
        if longest_streak == consecutive_coverage and consecutive_coverage > 0:
            start = current_streak_segments[0][0]
            end = current_streak_segments[-1][-1]
        
        if longest_streak > 0:
            starts.append(start)
            ends.append(end)
        else:
            starts.append(0)
            ends.append(0)
        
        coverage = longest_streak / num_segments
        coverage_scores.append(coverage)
        
    avg = np.mean(coverage_scores)
    std = np.std(coverage_scores)
    std_up = avg + std
        
    if save is not None:
        plot_coverage_mean(coverage_scores, lines, save=save)
        plot_coverage_lines(x, y, lines, coverage_scores, threshold=std_up, save=save)
    
    return coverage_scores, starts, ends

def make_new_lines(lines, coverages, starts, ends, threshold):
    """
    takes in the set of all lines and returns only the lines that passes the threshold
    threshold being the minimum coverage score needed

    Args:
        lines (list): the set of all lines that should theoretically cover all of the bounds
        coverages (int array): should be the same size as lines, and index should match as well. coverage scores made by calculate_line_coverages   
        starts (int array): the starts of where the lines are being covered
        ends (int array): the ends of where the lines are being covered
        threshold (int): the number of segments in a row that need to be reached for it to plausibly represent an arm

    Returns:
        new_lines (int array): just the lines that are sufficiently covered
        new_starts (int array): the starts of where the lines start being covered
        new_ends (int array): the ends of where the lines stop being covered
    """
    
    new_lines = []
    new_starts = []
    new_ends = []
    
    if threshold:
        for index, coverage in enumerate(coverages):
            if coverage > threshold:
                new_lines.append(lines[index])
                new_starts.append(starts[index])
                new_ends.append(ends[index])
    else: # include lines a standard deviation above
        mean = np.mean(coverages)
        std = np.std(coverages)
        cutoff = mean + std

        for index, coverage in enumerate(coverages):
            if coverage > cutoff:
                new_lines.append(lines[index])
                new_starts.append(starts[index])
                new_ends.append(ends[index])
    
    return new_lines, new_starts, new_ends

def calculate_intersection(line1, line2):
    """
    Takes two lines and check if they intersect, depending on what kind of line they are

    Args:
        line1 (int, int): (m, b) where m is slope and b is y intercept (or x for vertical lines)
        line2 (int, int): (m, b)

    Returns:
        (int, int): x and y coordinates of intersection
    """
    
    # Check if both lines are vertical
    if np.isinf(line1[0]) and np.isinf(line2[0]):
        return None  # No intersection if both are vertical
    
    # Check if both lines are horizontal
    elif np.abs(line1[0]) < 1e-10 and np.abs(line2[0]) < 1e-10:
        return None  # No intersection if both are horizontal
    
    # Check if one of the lines is vertical and the other is horizontal
    elif np.isinf(line1[0]) and np.abs(line2[0]) < 1e-10: # line1 vertical, line2 horizontal
        x = line1[1]
        y = line2[1]
        return (x, y)
    elif np.isinf(line2[0]) and np.abs(line1[0]) < 1e-10: # line2 vertical, line1 horizontal
        x = line2[1]
        y = line1[1]
        return (x, y)
    
    # Line1 is vertical and Line2 is neither
    elif np.isinf(line1[0]):
        x = line1[1]
        y = line2[0] * x + line2[1]
        return (x, y)
    
    # Line2 is vertical and Line1 is neither
    elif np.isinf(line2[0]):
        x = line2[1]
        y = line1[0] * x + line1[1]
        return (x, y)
    
    # Line1 is horizontal and Line2 is neither
    elif np.abs(line1[0]) < 1e-10:
        y = line1[1]
        if np.abs(line2[0]) > 1e-10:  # Ensure Line2 is not vertical
            x = (y - line2[1]) / line2[0]
            return (x, y)
    
    # Line2 is horizontal and Line1 is neither
    elif np.abs(line2[0]) < 1e-10:
        y = line2[1]
        if np.abs(line1[0]) > 1e-10:  # Ensure Line1 is not vertical
            x = (y - line1[1]) / line1[0]
            return (x, y)
    
    # Neither line is vertical or horizontal
    else:
        denom = (line1[0] - line2[0])
        if np.abs(denom) > 1e-10:  # Ensure lines are not parallel
            x = (line2[1] - line1[1]) / denom
            y = line1[0] * x + line1[1]
            return (x, y)
                
    return None

def is_point_in_segment(point, start, end, vertical=False):
    """
    checks if the intersection is inside the starts and ends of where the line is covered

    Args:
        point (int, int): (x, y)
        start (int): start of where the line is covered
        end (int): end of where the line is covered
        vertical (bool, optional): for if it is vertical, since i'm using x values to check. Defaults to False.

    Returns:
        bool: whether the intersection is within the start and end of coverage
    """
    
    x, y = point
    
    if vertical:
        point_inside = start <= y <= end
    else:
        point_inside = start <= x <= end
    
    return point_inside

def find_intersections(lines, starts, ends):
    """
    finds intersections between all the lines provided within bounds of starts and ends

    Args:
        lines (int, int): slope and y-intercept
        starts (int array): where the coverage of the line starts
        ends (int array): where the coverage of the line ends

    Returns:
        int array: list of x and y coordinates of where intersections are
    """
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = calculate_intersection(lines[i], lines[j])
            
            if intersection:
                # check if it is vertical, because start and end values are in y if so
                if np.isinf(lines[i][0]) and np.isinf(lines[j][0]):
                    if is_point_in_segment(intersection, starts[i], ends[i], vertical = True) and \
                       is_point_in_segment(intersection, starts[j], ends[j], vertical = True):
                        intersections.append(intersection)
                elif np.isinf(lines[i][0]):
                    if is_point_in_segment(intersection, starts[i], ends[i], vertical = True) and \
                       is_point_in_segment(intersection, starts[j], ends[j]):
                        intersections.append(intersection)
                elif np.isinf(lines[j][0]):
                    if is_point_in_segment(intersection, starts[i], ends[i]) and \
                       is_point_in_segment(intersection, starts[j], ends[j], vertical = True):
                        intersections.append(intersection)
                # no vertical lines
                elif is_point_in_segment(intersection, starts[i], ends[i]) and \
                     is_point_in_segment(intersection, starts[j], ends[j]):
                    intersections.append(intersection)
    
    return intersections

def make_convex_hull(intersection_points, eps=5, min_samples=30):
    #Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(intersection_points)

    # Find the cluster with the most points (highest concentration)
    cluster_indices, counts = np.unique(clusters, return_counts=True)
    densest_cluster_index = cluster_indices[np.argmax(counts)]
    densest_cluster_points = intersection_points[clusters == densest_cluster_index]
    
    convex_hull = ConvexHull(densest_cluster_points)
    return convex_hull, densest_cluster_points

def make_convex_hull_experimental(intersection_points, eps=5, min_samples=30, distance_threshold=5):
    """
    creates a convex hull around the intersection points found

    Args:
        intersection_points (np int array tuple): x and y coordinates of intersections

    Returns:
        scipy.spatial.ConvexHull: convex hull for the intersections
        np int array: all of the cluster points at the densest point
    """
    def check_segmentation(cluster_points, x_coord, y_coord, min_x, max_x, min_y, max_y, dist_threshold=30, num_seg=10):
        segment_len_x = (max_x - min_x) / num_seg
        segment_len_y = (max_y - min_y) / num_seg

        # check vertical line
        consecutive_coverage = 0
        longest_streak = 0
        total_coverage = 0
        multiple_segments_vertical = False
        for i in range(num_seg):
            segment_start = min_y + i * segment_len_y
            segment_end = segment_start + segment_len_y
            segment_points = cluster_points[(cluster_points[:, 1] >= segment_start) & (cluster_points[:, 1] <= segment_end)]
            
            if len(segment_points) > 0: # check if there are any points
                distances = np.abs(segment_points[:, 0] - x_coord)
                point_on_line = np.any(distances <= dist_threshold)
                    
                # add to consecutive coverage if there is something on line, no if not
                if point_on_line:
                    consecutive_coverage += 1
                    total_coverage += 1
                    if longest_streak < consecutive_coverage:
                        longest_streak = consecutive_coverage # keep track of longest streak of coverage
                    
                    if longest_streak > 1 and consecutive_coverage == 1:
                        multiple_segments_vertical = True
                else:
                    consecutive_coverage = 0
        coverage_vertical = total_coverage / num_seg
        
        # check horizontal line
        consecutive_coverage = 0
        longest_streak = 0
        total_coverage = 0
        multiple_segments_horizontal = False
        for i in range(num_seg):
            segment_start = min_x + i * segment_len_x
            segment_end = segment_start + segment_len_x
            segment_points = cluster_points[(cluster_points[:, 0] >= segment_start) & (cluster_points[:, 0] <= segment_end)]
            
            if len(segment_points) > 0: # check if there are any points
                distances = np.abs(segment_points[:, 1] - y_coord)
                point_on_line = np.any(distances <= dist_threshold)
                    
                # add to consecutive coverage if there is something on line, no if not
                if point_on_line:
                    consecutive_coverage += 1
                    total_coverage += 1
                    if longest_streak < consecutive_coverage:
                        longest_streak = consecutive_coverage # keep track of longest streak of coverage
                    
                    if longest_streak > 1 and consecutive_coverage == 1:
                        multiple_segments_horizontal = True
                else:
                    consecutive_coverage = 0
        coverage_horizontal = total_coverage / num_seg
        
        return multiple_segments_vertical, multiple_segments_horizontal, coverage_vertical, coverage_horizontal
    
    #Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(intersection_points)

    # Find the cluster with the most points (highest concentration)
    cluster_indices, counts = np.unique(clusters, return_counts=True)
    densest_cluster_index = cluster_indices[np.argmax(counts)]
    densest_cluster_points = intersection_points[clusters == densest_cluster_index]

    # Create a convex hull around the densest cluster
    point_outside = True
    iterated_points = densest_cluster_points
    concave_hull = alphashape.alphashape(densest_cluster_points, 0.1)
    buffer_zone = concave_hull.buffer(15)
    while point_outside:
        hull_polygon = sPolygon(iterated_points)
        min_x, min_y, max_x, max_y = hull_polygon.bounds
        
        point_found = False
        for x_coord in np.arange(min_x, max_x + 1, 1):
            if max_x < x_coord or min_x > x_coord:
                break

            for y_coord in np.arange(min_y, max_y + 1, 1):
                if max_y < y_coord or min_y > y_coord:
                    break
                
                point = Point(x_coord, y_coord)
                if buffer_zone.contains(point) == False:
                    point_found = True

                    # check which axis to reduce
                    max_x_dist = max_x - x_coord
                    min_x_dist = x_coord - min_x
                    max_y_dist = max_y - y_coord
                    min_y_dist = x_coord - min_y
                    
                    many_segs_v, many_segs_h, coverage_v, coverage_h = check_segmentation(densest_cluster_points, x_coord, y_coord, min_x, max_x, min_y, max_y)
                    
                    if many_segs_v and many_segs_h:
                        if max_x_dist > min_x_dist:
                            min_x = x_coord
                        else:
                            max_x = x_coord
                        
                        if max_y_dist > min_y_dist:
                            min_y = y_coord
                        else:
                            max_y = y_coord
                    elif many_segs_v:
                        if max_y_dist > min_y_dist:
                            min_y = y_coord
                        else:
                            max_y = y_coord
                    elif many_segs_h:
                        if max_x_dist > min_x_dist:
                            min_x = x_coord
                        else:
                            max_x = x_coord
                    """else:
                        if coverage_v > coverage_h:
                            if max_x_dist > min_x_dist:
                                min_x = x_coord
                            else:
                                max_x = x_coord
                        else:
                            if max_y_dist > min_y_dist:
                                min_y = y_coord
                            else:
                                max_y = y_coord"""

                    iterated_points = densest_cluster_points[
                        (densest_cluster_points[:, 1] <= max_y - 10) & (densest_cluster_points[:, 1] >= min_y + 10) &
                        (densest_cluster_points[:, 0] <= max_x - 10) & (densest_cluster_points[:, 0] >= min_x + 10)
                    ]
                
            if point_found:
                break

        if not point_found:
            point_outside = False
            break
    
    final_convex_hull = ConvexHull(iterated_points)
    
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], s=10, c='green')
    plt.scatter(densest_cluster_points[:, 0], densest_cluster_points[:, 1], s=10, c='red')

    # Plot concave hull
    if concave_hull:
        x_coord, y_coord = concave_hull.exterior.xy
        plt.plot(x_coord, y_coord, 'b-')

    # Plot final convex hull for the overlapping region
    for simplex in final_convex_hull.simplices:
        plt.plot(iterated_points[simplex, 0], iterated_points[simplex, 1], 'm-')

    plt.show()
    
    return final_convex_hull, densest_cluster_points


### PLOTTING METHODS -----------
def plot_lines(x, y, lines, title=None, save=None):
    """plots x & y coords on a backdrop of lines

    Args:
        x (list): float list
        y (list): float list
        lines (tuple): (m, b)
        title (str, optional): title if desired. Defaults to None.
        save (str, optional): file path of file. Defaults to None.
    """
    _, ax = plt.subplots()
    
    # plot data points
    ax.scatter(x, y, label = "Data Points", color="red")
    
    # get range
    x_min, x_max, y_min, y_max = calculate_range(x, y)
    
    # plotting lines
    for slope, b in lines:
        if np.isfinite(slope):
            if slope != 0:
                x_vals = np.array([x_min, x_max])
                y_vals = slope * x_vals + b

                ax.plot(x_vals, y_vals, "g--", linewidth=0.5, alpha=0.5)
            else: # horizontal lines
                ax.axhline(y=b, color="g", linestyle="--", linewidth=0.5)
        else: # vertical lines
            ax.axvline(x=b, color="g", linestyle="--", linewidth=0.5)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.grid(True)
    ax.set_aspect("equal", "box")
    ax.legend()
    
    if title:
        plt.title(title)
    
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

def plot_coverage_mean(coverage_scores, lines, save=None):
    """plotting to look at what the coverage mean + std looks like"""
    # plot distance to know what's a good threshold
    # get std & mean
    std = np.std(coverage_scores)
    mean = np.mean(coverage_scores)
    
    # plot - x is freq, y is avg dist
    plt.figure(figsize=(10, 6))
    plt.hist(coverage_scores, bins = len(lines), color="blue")
    plt.axvline(mean, color="r", linestyle="dashed", linewidth=2)
    plt.axvline(mean + std, color="g", linestyle="dashed", linewidth=2, label="1 std")
    plt.axvline(mean - std, color="g", linestyle="dashed", linewidth=2)
    
    plt.title("Coverage Score Distribution")
    plt.legend()
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

def plot_coverage_lines(x, y, lines, coverages, threshold=0.5, save=None):
    # filter for lines that align well
    new_lines = []
    
    for index, coverage in enumerate(coverages):
        if coverage > threshold:
            new_lines.append(lines[index])
    
    # plot
    plot_lines(x, y, new_lines, title = "New Lines", save=save)

def plot_segments(x, y, lines, starts, ends, save=None):
    plt.scatter(x, y)
    
    for i, (slope, b) in enumerate(lines):
        if np.isinf(slope) or np.abs(slope) > 1e10: # for vertical lines, their starts and ends are y coords
            start_y = starts[i]
            end_y = ends[i]
            
            if start_y == 0 and end_y == 0:
                continue
            else:
                start_x = b
                end_x = b

                plt.plot([start_x, end_x], [start_y, end_y], marker="o", color="r", linestyle="--")
            
        else: # horizontal & diagonal lines
            start_x = starts[i]
            end_x = ends[i]
        
            if start_x == 0 and end_x == 0:
                continue
            else:
                start_y = slope * start_x + b
                end_y = slope * end_x + b

                plt.plot([start_x, end_x], [start_y, end_y], marker="o", color="r", linestyle="--")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Segments")
    plt.grid(True)
    
    if save is not None:
        plt.savefig(f"{save}/segments.jpg")
    else:
        plt.show()

def plot_hull_with_intx_points(x, y, intersection_points, densest_cluster_points, hull, save=None):
    plt.scatter(x, y)

    # Plotting (optional, for visualization)
    plt.scatter(intersection_points[:,0], intersection_points[:,1], alpha=0.5, color="green")
    plt.scatter(densest_cluster_points[:,0], densest_cluster_points[:,1], color="red", alpha=0.3)
    for simplex in hull.simplices:
        plt.plot(densest_cluster_points[simplex, 0], densest_cluster_points[simplex, 1], "k-")

    # Create a Polygon patch for the convex hull
    hull_polygon = mPolygon(densest_cluster_points[hull.vertices], closed=True, edgecolor="k", fill=False)
    plt.gca().add_patch(hull_polygon)
    
    if save is not None:
        plt.savefig(f"{save}/convex_hull.jpg")
        plt.close()
    else:
        plt.show()

def plot_convex_hull(x, y, hull_indices):
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

## GET ZONES ------------------
def get_centre_hull(df, threshold=None, save_figures=None):
    """
    Creates the convex hull for centre zone of the maze for any given recording
    Methodology described above
    
    Args:
        x (int array): x coordinates for entire session
        y (int array): y coordinates for entire session
        save (str, optional): file path is saving is desired. Defaults to None
        threshold (int, optional): number of segments in a row needed to cross threshold when makign new lines. Defaults to None
        plot (bool, optional): whether to display plots
        
    Returns:
        (scipy.spatial.ConvexHull): convex hull corresponding to the centre zone
    """
    
    x = df["x"]
    y = df["y"]
    
    # file paths for saves
    save_path = os.path.join(helper.BASE_PATH, "processed_data", "hull_data")
    raw_path = os.path.join(save_path, f"{helper.CURRENT_RAT}_{helper.CURRENT_DAY}_raw_data.csv")
    covered_path = os.path.join(save_path, f"{helper.CURRENT_RAT}_{helper.CURRENT_DAY}_covered_lines.csv")
    intersections_path = os.path.join(save_path, f"{helper.CURRENT_RAT}_{helper.CURRENT_DAY}_intersections.csv")
    hull_path = os.path.join(save_path, f"{helper.CURRENT_RAT}_{helper.CURRENT_DAY}_hull.npy")
    
    # check if these files already exist
    if not(os.path.exists(hull_path)):
        save = True # saves regardless if file doesn't exist
    else:
        save = False
    
    if save:
        # step 1 - generate lines that cover the entire plot
        lines = create_lines(x, y, save=save_figures)
        
        # step 2 - calculate the coverage of how well the points cover the lines
        coverages, starts, ends = calculate_line_coverages(x, y, lines, save=save_figures)
        
        # step 3 - only keep the lines that are past the threshold
        updated_lines, updated_starts, updated_ends = make_new_lines(lines, coverages, starts, ends, threshold)
        if save_figures:
            plot_segments(x, y, updated_lines, updated_starts, updated_ends, save=save_path) # plot the new lines if desired
        
        # step 4 - find the intersection points between lines that still exist
        intersections = find_intersections(updated_lines, updated_starts, updated_ends)
        intersection_points = np.array(intersections) # np array for DBSCAN to work
        
        # step 5 - create convex hull
        hull, densest_cluster_points = make_convex_hull(intersection_points)
        if save_figures:
            plot_hull_with_intx_points(x, y, intersection_points, densest_cluster_points, hull, save=save_path)
        
        # separate lines into individual arrays
        slopes, b = zip(*lines)
        covered_slopes, covered_intercepts = zip(*updated_lines)
        
        # data frame
        raw_data = { # complete raw data
            "Coverages": coverages,
            "Slopes": slopes,
            "Intercepts": b,
            "Starts": starts,
            "Ends": ends
        }
        raw_df = pd.DataFrame(raw_data)
        
        covered_lines = {
            "Covered Slopes": covered_slopes,
            "Covered Intercepts": covered_intercepts
        }
        covered_df = pd.DataFrame(covered_lines)
        
        intersections_df = pd.DataFrame(intersections, columns=["Intersections X", "Intersections Y"])
        
        # save
        raw_df.to_csv(raw_path)
        covered_df.to_csv(covered_path)
        intersections_df.to_csv(intersections_path)
        
        # save convex hull
        np.save(hull_path, densest_cluster_points[hull.vertices])
    else:
        # load hull
        densest_cluster_points = np.load(hull_path)
        hull = ConvexHull(densest_cluster_points)
        #hull_vertices = hull.vertices
        #plot_convex_hull(df["x"], df["y"], hull_vertices)
    
    return hull