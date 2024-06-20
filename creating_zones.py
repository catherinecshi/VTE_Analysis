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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon


def calculate_range(x, y):
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    
    return x_min, x_max, y_min, y_max

def generate_lines(x, y, gap_between_lines = 20, degree_step = 10, min_length = 950, hv_line_multiplier = 2, plot = False):
    """
    Generate a set of lines within bounds (the x and y coordinates that bound the points in dataframe) that cover most of the bounds

    Args:
        x (array): x coords from df
        y (array): y coords from df
        gap_between_lines (int, optional): the increment between x/y intercepts as the set of lines are created. Defaults to 20.
        degree_step (int, optional): the increment between the degree of the slope for the lines. Defaults to 10.
        min_length (int, optional): the minimum length of lines to be included in the final set of lines. Done to avoid corner lines with short ass segments. Defaults to 950.
        hv_line_multiplier (int, optional): multiplier for how many horizontal/vertical lines compared to angled lines. Defaults to 2.
        plot (bool, optional): whether to plot the lines, mostly to check if they've been generated in a way to covers the bounds well

    Returns:
        list: list of tuples representing the line. includes the slope (m) & y-intercept (b) for angled lines
        
    Procedure:
        1. create an array of possible slopes by incrementing using degree_step
        2. determine the number of lines needed
            - since the lines are incremented based on x intercept, how many lines determine on the range it has to cover
        3. for each slope
            - calculate the number of lines that can be created based on x intercept, incrementing by gap_between_lines
            - get y intercept for each possible x intercept
            - discard short lines, as determined by length of line within the bounds
            - then do the same based on incrementing the y intercept
        4. for horizontal lines
            - increment based on y-intercept (b)
            - slope is 0
        5. for vertical lines
            - increment based on x-intercept
            - slope is np.inf and b is the x-intercept instead
    """
    
    lines = []
    
    # convert degree steps into slopes where m = tan(angle in radians)
    angle_degrees = np.arange(0, 180, degree_step)
    slopes = np.tan(np.radians(angle_degrees))
    
    # get range
    x_min, x_max, y_min, y_max = calculate_range(x, y)
    extended_range = max(x_max - x_min, y_max - y_min) * 1.5
    
    # determine number of lines needed
    num_angled_lines = int(2 * extended_range / gap_between_lines) + 1
    num_hv_lines = num_angled_lines * hv_line_multiplier
    
    for slope in slopes:
        if np.isfinite(slope) and np.abs(slope) < 1e10 and slope != 0:
            # increment based on x intercept - add gap incrementally
            # get number of steps i can take
            x_steps = (x_max - x_min) / gap_between_lines
            x_steps = math.ceil(x_steps)
            
            for i in range(x_steps):
                # calculate x intercept
                x_intercept = (i * gap_between_lines) + x_min
                
                # get b value (y intercept) -> b = y - mx
                b = 0 - (slope * x_intercept) + x_min
                
                # check for length of line and discard short ones
                # determine end points within range
                y_at_x_min = slope * x_min + b
                y_at_x_max = slope * x_max + b
                x_at_y_min = (y_min - b) / slope
                x_at_y_max = (y_max - b) / slope

                # clip line into range
                start_x = max(min(x_at_y_min, x_at_y_max, x_max), x_min)
                end_x = min(max(x_at_y_min, x_at_y_max, x_min), x_max)
                start_y = max(min(y_at_x_min, y_at_x_max, y_max), y_min)
                end_y = min(max(y_at_x_min, y_at_x_max, y_min), y_max)
                
                # calculate length of lines
                length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                if length >= min_length: # only include line if it is long enough
                    lines.append((slope ,b))
            
            # same thing but now increment based on y intercept
            y_steps = (y_max - y_min) / gap_between_lines
            y_steps = math.ceil(y_steps)
            
            for i in range(y_steps):
                b = (i * gap_between_lines) + y_min
                
                # check for length of line and discard short ones
                # determine end points within range
                y_at_x_min = slope * x_min + b
                y_at_x_max = slope * x_max + b
                x_at_y_min = (y_min - b) / slope
                x_at_y_max = (y_max - b) / slope

                # clip line into range
                start_x = max(min(x_at_y_min, x_at_y_max, x_max), x_min)
                end_x = min(max(x_at_y_min, x_at_y_max, x_min), x_max)
                start_y = max(min(y_at_x_min, y_at_x_max, y_max), y_min)
                end_y = min(max(y_at_x_min, y_at_x_max, y_min), y_max)
                
                # calculate length of lines
                length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                if length >= min_length: # only include line if it is long enough
                    lines.append((slope ,b))
                
    # generating horizontal and vertical lines - all long enough so no need for filtering
    # horizontal lines
    for i in range(-num_hv_lines // 2, num_hv_lines // 2):
        b = y_min + (i * gap_between_lines / hv_line_multiplier) # have more horizontal and vertical lines
        lines.append((0, b))
        
    #vertical lines
    for i in range(-num_hv_lines // 2, num_hv_lines // 2):
        x_position = x_min + (i * gap_between_lines / hv_line_multiplier)
        lines.append((np.inf, x_position))
    
    if plot:
        plot_lines(x, y, lines, title = "Original Lines")
            
    return lines

def calculate_line_coverages(x, y, lines, num_segments = 15, threshold = 5, plot = False):
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
        
    if plot:
        plot_coverage_mean(coverage_scores, lines)
        plot_coverage_lines(x, y, lines, coverage_scores)
    
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

def is_point_in_segment(point, start, end, vertical = False):
    """
    checks if the intersection is inside the starts and ends of where the line is covered

    Args:
        point (int, int): x and y coordinates
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

"""
def make_ellipse(points, ax = None, scale_factor = 1.5):
    # Fit the DBSCAN clusterer to the points
    clustering = DBSCAN(eps=30, min_samples=10).fit(points)  # Adjust eps and min_samples as needed
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters_ > 0:
        # Proceed with PCA and ellipse drawing for the largest cluster
        # Find the largest cluster
        largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
        cluster_points = points[labels == largest_cluster_idx]
        
        # PCA to determine the orientation
        pca = PCA(n_components=2).fit(cluster_points)
        center = pca.mean_
        angle = np.arctan2(*pca.components_[0][::-1]) * (180 / np.pi)
        width, height = 2 * np.sqrt(pca.explained_variance_) 
        
        # make ellipse bigger
        width *= scale_factor
        height *= scale_factor

        return {'center': center, 'width': width, 'height': height, 'angle': angle}
    else:
        return None
"""

def make_convex_hull(intersection_points):
    """
    creates a convex hull around the intersection points found

    Args:
        intersection_points (np int array tuple): x and y coordinates of intersections

    Returns:
        scipy.spatial.ConvexHull: convex hull for the intersections
        np int array: all of the cluster points at the densest point
    """
    
    #Perform DBSCAN clustering
    dbscan = DBSCAN(eps=10, min_samples=5)  # Adjust these parameters as necessary
    clusters = dbscan.fit_predict(intersection_points)

    # Find the cluster with the most points (highest concentration)
    cluster_indices, counts = np.unique(clusters, return_counts=True)
    densest_cluster_index = cluster_indices[np.argmax(counts)]
    densest_cluster_points = intersection_points[clusters == densest_cluster_index]

    # Create a convex hull around the densest cluster
    hull = ConvexHull(densest_cluster_points)
    
    return hull, densest_cluster_points


### PLOTTING METHODS -----------
def plot_lines(x, y, lines, title):
    fig, ax = plt.subplots()
    
    # plot data points
    ax.scatter(x, y, label = "Data Points")
    
    # get range
    x_min, x_max, y_min, y_max = calculate_range(x, y)
    
    # plotting lines
    for slope, b in lines:
        if np.isfinite(slope):
            if slope != 0:
                x_vals = np.array([x_min, x_max])
                y_vals = slope * x_vals + b

                ax.plot(x_vals, y_vals, 'r--', linewidth = 0.5)
            else: # horizontal lines
                ax.axhline(y=b, color = 'r', linestyle = '--', linewidth = 0.5)
        else: # vertical lines
            ax.axvline(x=b, color = 'r', linestyle = '--', linewidth = 0.5)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    ax.legend()
    
    if title:
        plt.title(title)
    
    plt.show()

def plot_coverage_mean(coverage_scores, lines):
    # plot distance to know what's a good threshold
    # get std & mean
    std = np.std(coverage_scores)
    mean = np.mean(coverage_scores)
    
    # plot - x is freq, y is avg dist
    plt.figure(figsize=(10, 6))
    plt.hist(coverage_scores, bins = len(lines), color = 'skyblue')
    plt.axvline(mean, color = 'r', linestyle = 'dashed', linewidth = 2)
    plt.axvline(mean + std, color = 'g', linestyle = 'dashed', linewidth = 2, label = '1 std')
    plt.axvline(mean - std, color = 'g', linestyle = 'dashed', linewidth = 2)
    
    plt.title("Coverage Score Distribution")
    plt.legend()
    plt.show()

def plot_coverage_lines(x, y, lines, coverages, threshold = 0.8):
    # filter for lines that align well
    new_lines = []
    
    for index, coverage in enumerate(coverages):
        if coverage > threshold:
            #print(index)
            new_lines.append(lines[index])
    
    # plot
    plot_lines(x, y, new_lines, title = "New Lines")

def plot_segments(x, y, lines, starts, ends, ratID = 'Misc', day = None):
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

                plt.plot([start_x, end_x], [start_y, end_y], marker = 'o', color = 'r', linestyle = '--')
            
        else: # horizontal & diagonal lines
            start_x = starts[i]
            end_x = ends[i]
        
            if start_x == 0 and end_x == 0:
                continue
            else:
                start_y = slope * start_x + b
                end_y = slope * end_x + b

                plt.plot([start_x, end_x], [start_y, end_y], marker = 'o', color = 'r', linestyle = '--')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Segments")
    plt.grid(True)
    plt.show()
    
    # save the figure
    if day:
        plt.savefig(f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/{ratID}/{day}')

def plot_hull(x, y, intersection_points, densest_cluster_points, hull, day = None):
    plt.scatter(x, y)

    # Plotting (optional, for visualization)
    plt.scatter(intersection_points[:,0], intersection_points[:,1], alpha=0.5, color = 'green')
    plt.scatter(densest_cluster_points[:,0], densest_cluster_points[:,1], color='red')
    for simplex in hull.simplices:
        plt.plot(densest_cluster_points[simplex, 0], densest_cluster_points[simplex, 1], 'k-')

    # Create a Polygon patch for the convex hull
    hull_polygon = Polygon(densest_cluster_points[hull.vertices], closed=True, edgecolor='k', fill=False)
    plt.gca().add_patch(hull_polygon)
    
    if day:
        plt.savefig(f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13/{day}')
    else:
        plt.show()


## GET ZONES ------------------
def get_centre_zone(x, y, ratID, day, save = False, threshold = None, plot = False): #currently highly experimental, ask cat for an exp if needed
    # determine whether saving is necessary
    file_path = f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/{ratID}/{day}'
    coverage_path = file_path + '/coverage_scores.csv'
    covered_path = file_path + '/covered_lines.csv'
    intersections_path = file_path + '/intersections.csv'
    hull_path = file_path + '/hull_vertices.npy'
    
    if save or not os.path.exists(coverage_path) or not os.path.exists(hull_path): # saves regardless if file doesn't exist
        # step 1 - generate lines that cover the entire plot
        lines = generate_lines(x, y, plot = plot)
        
        # step 2 - calculate the coverage of how well the points cover the lines
        coverages, starts, ends = calculate_line_coverages(x, y, lines)
        
        # step 3 - only keep the lines that are past the threshold
        updated_lines, updated_starts, updated_ends = make_new_lines(lines, coverages, starts, ends, threshold)
        if plot:
            plot_segments(x, y, updated_lines, updated_starts, updated_ends, ratID = ratID, day = day) # plot the new lines if desired
        
        # step 4 - find the intersection points between lines that still exist
        intersections = find_intersections(updated_lines, updated_starts, updated_ends)
        intersection_points = np.array(intersections) # np array for DBSCAN to work
        
        # step 5 - create convex hull
        hull, densest_cluster_points = make_convex_hull(intersection_points, x, y, day = day)
        if plot:
            plot_hull(x, y, intersection_points, densest_cluster_points, hull, day)
        
        # separate lines into individual arrays
        slopes, b = zip(*lines)
        covered_slopes, covered_intercepts = zip(*updated_lines)
        
        # data frame
        df = pd.DataFrame(coverages, columns=['Coverages'])
        df['Slope'] = slopes # should be indexed the same as coverages
        df['Intercept'] = b
        df['Starts'] = starts
        df['Ends'] = ends
        
        # for the filtered processed data
        df2 = pd.DataFrame(covered_slopes, columns=['Covered Slopes'])
        df2['Covered Intercepts'] = covered_intercepts
        df3 = pd.DataFrame(intersections, columns=['Intersections X', 'Intersections Y'])

        # save
        df.to_csv(coverage_path)
        df2.to_csv(covered_path)
        df3.to_csv(intersections_path)
        
        # save convex hull
        np.save(hull_path, densest_cluster_points[hull.vertices])
    else:
        # probably not necessary right now?
        """df = pd.read_csv(file_path)
        slopes = df['Slope'].tolist()
        b = df['Intercept'].tolist()
        coverages = df['Coverages'].tolist()
        starts = df['Starts'].tolist()
        ends = df['Ends'].tolist()"""
        
        # load hull
        densest_cluster_points = np.load(hull_path)
        hull = ConvexHull(densest_cluster_points)
        
        # remake lines if needed
        #lines = list(zip(slopes, b))
    
    return hull