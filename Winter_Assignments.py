# packages
import os
import pickle
import bisect
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import zscore

# other helper files
import data_structure
import performance_analysis

# DATA MANIPULATION/ANALYSIS METHODS ---------
def filter_dataframe(df, track_part = 'greenLED', std_multiplier = 7, eps = 70, min_samples = 40, distance_threshold = 190): # currently keeps original indices
    # modify a copy instead of the original
    # also filter based on likelihood values
    likely_data = df[df[(track_part, 'likelihood')] > 0.999].copy()
    
    # DBSCAN Cluster analysis
    coordinates = likely_data[[track_part]].copy()[[(track_part, 'x'), (track_part, 'y')]]
    coordinates.dropna(inplace = True) # don't drop nan for dbscan
    
    clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(coordinates)
    labels = clustering.labels_
    #noise_points_count = (labels == -1).sum() # so ik how many points were filtered out
    #print(f"DBSCAN Filtered out {noise_points_count}")

    filtered_indices = labels != -1 # filter out noise
    filtered_data = likely_data[filtered_indices].copy()
    
    # calculate thresholds
    diff_x = df[(track_part, 'x')].diff().abs()
    diff_y = df[(track_part, 'y')].diff().abs()
    threshold_x = diff_x.std() * std_multiplier
    threshold_y = diff_y.std() * std_multiplier
    
    # calculate diff between current point and last non-jump point
    last_valid_index = 0
    jump_indices = [] # just to see how many points are jumped over
    
    for i in range(1, len(filtered_data)):
        diff_x = abs(filtered_data.iloc[i][(track_part, 'x')] - filtered_data.iloc[last_valid_index][(track_part, 'x')])
        diff_y = abs(filtered_data.iloc[i][(track_part, 'y')] - filtered_data.iloc[last_valid_index][(track_part, 'y')])
        #distance = np.sqrt(diff_x**2 + diff_y**2) # euclidean distance
        
        # check for jumps
        if diff_x > threshold_x or diff_y > threshold_y:
            # mark as NaN
            filtered_data.at[filtered_data.index[i], (track_part, 'x')] = np.nan
            filtered_data.at[filtered_data.index[i], (track_part, 'y')] = np.nan
            jump_indices.append(i)
        else:
            # udpate last valid index
            last_valid_index = i
    
    # interpolating
    filtered_data[(track_part, 'x')].interpolate(inplace = True)
    filtered_data[(track_part, 'y')].interpolate(inplace = True)
    
    print(f"number of points filtered out - {len(jump_indices)}")
    
    # final coordinate points
    x = filtered_data[(track_part, 'x')]
    y = filtered_data[(track_part, 'y')]
    
    return x, y

def get_time(content, statescript_time):
    lines = content.splitlines()
    
    starting_time = None # store the starting time
    
    for line in lines:
        if '#' in line:
            continue # skip the starting comments
        elif all(char.isspace() or char.isdigit()for char in line):
            parts = line.split()
            starting_time = int(parts[0])
            break
    
    # calculating real time passed since start of session
    time_passed = statescript_time - starting_time
    time_passed = time_passed / 1000 # turning it from ms to seconds
    
    return time_passed
            
def calculate_framerate(content, x):
    # get the last line in the statescript log that only has numbers
    last_line = None
    
    lines = content.splitlines()
    for line in lines:
        if all(char.isdigit() or char.isspace() for char in line):
            last_line = line
    
    # get the time value from the line
    time = None
    
    for index, char in enumerate(last_line):
        if char.isspace():
            time = last_line[:index]
            break
    
    # turn from string to integer
    time = int(time)
    
    # turn into seconds
    get_time(content, time)
    
    # calculate framerate
    framerate = time / len(x)
    
    return framerate

def calculate_speed(x, y, framerate):
    diff_x = x.diff().fillna(0)
    diff_y = y.diff().fillna(0)
    
    displacement_per_frame = np.sqrt(diff_x**2 + diff_y**2)
    
    speed = displacement_per_frame * framerate
    
    return speed            
 
def time_until_choice(content): # currently only for the first choice
    lines = content.splitlines()
    
    # some variables
    new_trial = False # to check if a trial is in session
    time = []
    arm = set()
    lick_count = 0
    not_first = False
    time_home = None
    time_arm = None
    
    for line in lines:
        if '#' in line: # skip starting comments
            continue
        elif 'New Trial' in line:
            new_trial = True
        elif all(char.isdigit() or char.isspace() for char in line) and new_trial: # a normal licking line
            parts = line.split()
            current_arm = int(parts[1])
            current_time = int(parts[0])
            
            # check when rat left home
            if current_arm > 1 and not not_first: # licked at arm not at home
                time_home = time[-1] # haven't appended to time yet so this is last line's time
                time_arm = current_time
            
            # check for how many licks
            if current_arm > 1:
                if current_arm in arm: # not a new arm
                    lick_count += 1
                else:
                    lick_count = 1 # new arm so reset lick count
            
            # check if the rat changed its choice without licking enough at one arm
            if current_arm > 1 and len(arm) > 3:
                time_arm = current_time
            
            time.append(current_time)
            arm.add(current_arm)

        if lick_count == 3: # enough licks for a choice
            break

    # calculating the difference between the time
    time_arm_seconds = get_time(content, time_arm)
    time_home_seconds = get_time(content, time_home)
    time_diff = time_arm_seconds - time_home_seconds
    
    return time_diff

def check_timestamps(df, timestamps):
    # first check - makes sure there is around 0.03s between each frame
    time_off = np.zeros(len(timestamps)) # records indices where time isn't ~0.03s between frames
    index_off = 0
    print(timestamps)
    
    for index, time in enumerate(timestamps): #loops through timestamps
        if index == 0:
            continue
    
        # calculate diff in seconds between frames
        current_time = timestamps[index]
        past_time = timestamps[index - 1]
        time_diff = current_time - past_time
        
        # make sure it's around the 0.03 range
        if time_diff > 0.05 or time_diff < 0.01:
            print(time_diff)
            time_off[index] = time_diff # time is off here
            
            if index_off < 5:
                print(f"time_diff is off norm for {index}")
                #index_off += 1
        else:
            continue
    
    # second check - make sure x and timestamps are the same length
    if not(len(df) == len(timestamps)):
        print("length of x and timestamps don't match up")
        print(len(df))
        print(len(timestamps))
        
        diff = len(df) - len(timestamps)
        # it seems like most of them differ by 1, where df = timestamps - 1, so i'm doing a rough subtraction here
        if diff == 1:
            timestamps.pop()
    
    return timestamps

def calculate_range(x, y):
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    
    return x_min, x_max, y_min, y_max

def is_point_in_ellipse(x, y, ellipse_params):
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


# VTE METHODS -------------
def define_zones(x, y, x_min, x_max, y_min, y_max, will_plot = False):
    # calculate elliptical parameters
    centre_x = (x_min + x_max) / 2
    centre_y = (y_min + y_max) / 2
    radius_x = (x_max - x_min) / 2
    radius_y = (y_max - y_min) / 2
    
    distances = ((x - centre_x) / radius_x) ** 2 + ((y - centre_y) / radius_y) ** 2 # distance from centre for each point
    
    # confine to bounds
    region_mask = distances <= 1
    filtered_points = np.array(list(zip(x[region_mask], y[region_mask]))) # combine into single array

    # convex hull
    hull = ConvexHull(filtered_points)
    
    # plot if wanted
    if will_plot:
        plot_hull(x, y, filtered_points, hull, centre_x, centre_y, radius_x, radius_y)
    
    return hull

def check_if_inside(point, hull):
    # delaunay triangulation of hull points
    del_tri = Delaunay(hull.points)
    
    # check if point is inside the hull
    return del_tri.find_simplex(point) >= 0

def get_trial_start_times(timestamps, SS_df): # gets indices for x/y where trials start & corresponding trial type
    lines = SS_df.splitlines()
    
    # storage variables
    start_of_trial = False # know when the last line was the start of new trial
    trial_starts = []
    trial_info = {} # store trial start times and trial types
    
    # get the trial start times from SS
    for line in lines:
        if line.startswith('#'): # skip the starting comments
            continue
        
        elif start_of_trial and "trialType" in line: # store trial type
            parts = line.split()
            trial_type = parts[3]
            trial_info[trial_start[-1]] = trial_type # assumes this will always come after New Trial'
                  
        elif 'New Trial' in line: # indicate start of a new trial
            start_of_trial = True
            
            # store the time during this event
            parts = line.split()
            trial_start = parts[0]
            trial_info[trial_start] = None
            trial_starts.append(trial_start)
        
        else:
            start_of_trial = False
    
    video_starts = video_trial_starts(timestamps, trial_starts) # this should be the indices for x/y where trials start
    
    # change trial_info such that the key is video_starts instead of trial_starts
    video_trial_info = {}
    
    if len(video_starts) == len(trial_starts):
        for index, video_start in enumerate(video_starts):
            original_start_time = trial_starts[index]
            trial_type = trial_info.get(original_start_time)
            video_trial_info[video_start] = trial_type
    
    return video_trial_info

def video_trial_starts(timestamps, SS_times):
    trial_starts = []
    
    for time in SS_times:
        # ensure consistent data types
        time = float(int(time) / 1000)
        
        if time in timestamps: # if there is a perfect match between MCU & ECU and the time is in timestamps
            index = timestamps.index(time)
            trial_starts.append(timestamps[index])
        else: # if there isn't a perfect match
            print(f"Imperfect match between ECU and MCU at {time}")
            
            # index where time is inserted into timestamps
            idx = bisect.bisect_left(timestamps, time)
            
            # check neighbours for closest time
            if idx == 0:
                trial_starts.append(timestamps[0])
            elif idx == len(timestamps):
                trial_starts.append(timestamps[-1])
            else:
                before = timestamps[idx - 1]
                after = timestamps[idx]
                closest_time = before if (time - before) <= (after - time) else after
                trial_starts.append(closest_time)
    
    return trial_starts

def DBSCAN_window(x, y): # failed. tried to use DBSCAN to determine central choice point window
    # the centre is probably gonna be in this range
    x_min, x_max = 450, 700
    y_min, y_max = 330, 570
    
    # filter to only include the central range
    central_region_mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    central_points = np.array(list(zip(x[central_region_mask], y[central_region_mask]))) # combine into single array
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps = 10, min_samples = 100) # where eps -> radius around point & min -> min points to be dense
    clusters = dbscan.fit_predict(central_points)
    
    # identify central cluster by looking for densest cluster
    if len(set(clusters)) > 1: # checks to make sure a cluster has been identified
        central_cluster_label = max(set(clusters), key = list(clusters).count)
        cluster_points = central_points[clusters == central_cluster_label]
    else:
        cluster_points = central_points
    
    # apply convex hull -> so end result isn't just a rectangular space
    if len(cluster_points) > 3:
        hull = ConvexHull(central_points)
        
        # plotting
        plt.scatter(x, y, alpha = 0.5)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color = 'red')
        #for simplex in hull.simplices:
            #plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-')
            
        plt.show()
    else: 
        print("Not enough points for convex hull in the central cluster")
    
    return

def calculate_trajectory(x, y, window_size = 100):
    # Assuming x and y are your coordinates
    x_median = np.median(x)
    y_median = np.median(y)

    # Define a window size based on your observations of the maze's layout
    window_size = 100  # This is an example size; adjust based on your specific maze dimensions

    # Define the choice point window around the medians
    window_bounds = {
        'xmin': x_median - window_size / 2,
        'xmax': x_median + window_size / 2,
        'ymin': y_median - window_size / 2,
        'ymax': y_median + window_size / 2
    }

    # Plot to verify the window
    plt.scatter(x, y, alpha=0.5)  # Plot all points
    plt.gca().add_patch(plt.Rectangle((window_bounds['xmin'], window_bounds['ymin']), window_size, window_size, linewidth=1, edgecolor='r', facecolor='none'))
    plt.axvline(x=x_median, color='k', linestyle='--')
    plt.axhline(y=y_median, color='k', linestyle='--')
    plt.title('Estimated Choice Point Area')
    plt.show()

def derivative(values, sr, d, m): # assumes each value is separated by regular time intervals -> sr
    v_est = np.zeros_like(values) # initialise slope array with zeroes / velocity estimates
    print(values)
    
    # start from second element for differentiation
    for i in range(1, len(values)):
        window_len = 0
        can_increase_window = True

        while True: # infinite loop
            window_len += 1
            
            if window_len > m or i - window_len < 0: # reached end of window / safety check
                window_len -= 1
                break
            
            # calculate slope from values[i] to values[i - window_len]
            slope_ = v_est[i - 1] # save previous slope / i changed from original code to be v_est[i - 1] instead of v_est[i]
            slope = (values[i] - values[i - window_len]) / (window_len * sr)
            
            if window_len > 1:
                print("window_len > 1")
                # y = mx + c where c -> y-intercept, values[i] -> y, slope -> m, i * sr -> x (time at point i)
                c = values[i] - slope * i * sr

                # check every point
                for j in range(1, window_len):
                    # diff between actual point and position calculated by model at every point up to i
                    delta = values[i - j] - (c + slope * (i - j) * sr)
                    
                    # use delta to assess quality of linear approximation -> 2 * d is threshold
                    if abs(delta) > 2 * d: # if model too far from actuality, excludes the problematic point in model
                        can_increase_window = False
                        window_len -= 1
                        slope = slope_
                        print("model too far from actual results")
                        break
            
            if not can_increase_window:
                break # exit while loop if window cannot be increased
        
        v_est[i] = slope
    
    return v_est
            
def calculate_IdPhi(trajectory_x, trajectory_y):
    # parameters - need to change
    sr = 0.02 # sampling rate
    d = 0.05 # position noise boundary
    m = 20 # window size
    
    # derivatives
    dx = derivative(trajectory_x, sr, d, m)
    print("got derivative for x")
    dy = derivative(trajectory_y, sr, d, m)
    print("got derivative for y")
    
    # calculate + unwrap angular velocity
    Phi = np.arctan2(dy, dx)
    Phi = np.unwrap(Phi)
    dPhi = derivative(Phi, sr, d, m)
    print("got derivative for Phi")
    
    # integrate change in angular velocity
    IdPhi = np.trapz(np.abs(dPhi))
    
    print("returning IdPhi")
    return IdPhi
   
            
# GETTING ZONES -----------
def generate_lines(x, y, gap_between_lines = 20, degree_step = 10, min_length = 950):
    lines = []
    
    # convert degree steps into slopes where m = tan(angle in radians)
    angle_degrees = np.arange(0, 180, degree_step)
    slopes = np.tan(np.radians(angle_degrees))
    
    # get range
    x_min, x_max, y_min, y_max = calculate_range(x, y)
    extended_range = max(x_max - x_min, y_max - y_min) * 1.5
    
    # determine number of lines needed
    num_lines = int(2 * extended_range / gap_between_lines) + 1
    
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
    for i in range(-num_lines // 2, num_lines // 2):
        b = y_min + (i * gap_between_lines)
        lines.append((0, b))
        
    #vertical lines
    for i in range(-num_lines // 2, num_lines // 2):
        x_position = x_min + (i * gap_between_lines)
        lines.append((np.inf, x_position))
            
    return lines

def calculate_line_coverages(x, y, lines, num_segments = 15, threshold = 5):
    coverage_scores = []
    points = np.array(list(zip(x, y)))
    x_min, x_max, y_min, y_max = calculate_range(x, y)
    
    for slope, b in lines:
        consecutive_coverage = 0 # what i'm looking for is consecutive coverage, much like a line
        longest_streak = 0
        
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
                        consecutive_coverage += 1
                        if longest_streak < consecutive_coverage:
                            longest_streak = consecutive_coverage # keep track of longest streak of coverage
                    else:
                        consecutive_coverage = 0
            
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
                        consecutive_coverage += 1
                        if longest_streak < consecutive_coverage:
                            longest_streak = consecutive_coverage # keep track of longest streak of coverage
                    else:
                        consecutive_coverage = 0
            
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
        
        segment_coverages = []
        for i in range(num_segments):
            segment_start = x_min + i * segment_length
            segment_end = segment_start + segment_length
            segment_points = points[(x >= segment_start) & (x <= segment_end)]
            
            if len(segment_points) > 0:
                distances = np.abs(slope * segment_points[:, 0] - segment_points[:, 1] + b) / np.sqrt(slope ** 2 + 1)
                point_on_line = np.any(distances <= threshold)
                    
                # add to consecutive coverage if there is something on line, no if not
                if point_on_line:
                    consecutive_coverage += 1
                    if longest_streak < consecutive_coverage:
                        longest_streak = consecutive_coverage # keep track of longest streak of coverage
                else:
                    consecutive_coverage = 0
        
        coverage = longest_streak / num_segments
        coverage_scores.append(coverage)
    
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
    
    plt.legend()
    plt.show()
    
    return coverage_scores

def make_new_lines(lines, coverages, threshold):
    new_lines = []
    
    for index, coverage in enumerate(coverages):
        if coverage > threshold:
            #print(index)
            new_lines.append(lines[index])
    
    return new_lines

def find_intersections(lines):
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            # Check if both lines are vertical
            if np.isinf(line1[0]) and np.isinf(line2[0]):
                continue  # No intersection if both are vertical
            
            # Check if both lines are horizontal
            elif np.abs(line1[0]) < 1e-10 and np.abs(line2[0]) < 1e-10:
                continue  # No intersection if both are horizontal
            
            # Check if one of the lines is vertical and the other is horizontal
            elif np.isinf(line1[0]) and np.abs(line2[0]) < 1e-10: # line1 vertical, line2 horizontal
                x = line1[1]
                y = line2[1]
                intersections.append((x, y))
            elif np.isinf(line2[0]) and np.abs(line1[0]) < 1e-10: # line2 vertical, line1 horizontal
                x = line2[1]
                y = line1[1]
                intersections.append((x, y))
            
            # Line1 is vertical and Line2 is neither
            elif np.isinf(line1[0]):
                x = line1[1]
                y = line2[0] * x + line2[1]
                intersections.append((x, y))
            
            # Line2 is vertical and Line1 is neither
            elif np.isinf(line2[0]):
                x = line2[1]
                y = line1[0] * x + line1[1]
                intersections.append((x, y))
            
            # Line1 is horizontal and Line2 is neither
            elif np.abs(line1[0]) < 1e-10:
                y = line1[1]
                if np.abs(line2[0]) > 1e-10:  # Ensure Line2 is not vertical
                    x = (y - line2[1]) / line2[0]
                    intersections.append((x, y))
            
            # Line2 is horizontal and Line1 is neither
            elif np.abs(line2[0]) < 1e-10:
                y = line2[1]
                if np.abs(line1[0]) > 1e-10:  # Ensure Line1 is not vertical
                    x = (y - line1[1]) / line1[0]
                    intersections.append((x, y))
            
            # Neither line is vertical or horizontal
            else:
                denom = (line1[0] - line2[0])
                if np.abs(denom) > 1e-10:  # Ensure lines are not parallel
                    x = (line2[1] - line1[1]) / denom
                    y = line1[0] * x + line1[1]
                    intersections.append((x, y))
                
    return intersections

def make_ellipse(points, ax = None, scale_factor = 3):
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


# PLOTTING METHODS --------
def create_scatter_plot(x, y):
    plt.figure(figsize = (10, 6))
    #plt.scatter(x, y, c = 'green', alpha = 0.6)
    plt.plot(x, y, color='green', alpha=0.4)
    plt.title('Tracking Data')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.show()

def create_occupancy_map(x, y, framerate, bin_size = 15):
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
    
    # plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(occupancy_grid, cmap=custom_cmap, interpolation='nearest')
    plt.colorbar(label='Time spent in seconds')
    plt.title('Occupancy Map')
    plt.xlabel('X Bins')
    plt.ylabel('Y Bins')
    plt.show()

def plot_hull(x, y, filtered_points, hull, centre_x, centre_y, radius_x, radius_y):
    plt.plot(x, y, 'o', markersize=5, label='Outside')
    plt.plot(filtered_points[:, 0], filtered_points[:, 1], 'ro', label='inside')
    
    for simplex in hull.simplices:
        plt.plot(filtered_points[simplex, 0], filtered_points[simplex, 1], 'k-')
        
    plt.fill(filtered_points[hull.vertices, 0], filtered_points[hull.vertices, 1], 'k', alpha = 0.1)
    ellipse = Ellipse((centre_x, centre_y), width = 2 * radius_x, height = 2 * radius_y, edgecolor = 'g', fill = False, linewidth = 2, linestyle = '--')
    plt.gca().add_artist(ellipse)
    
    plt.legend()
    plt.show()

def plot_lines(x, y, lines):
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
    plt.show()

def plot_coverages(x, y, lines, coverages, threshold = 0.8):
    # filter for lines that align well
    new_lines = []
    #print(len(coverages))
    #print(len(lines))
    
    for index, coverage in enumerate(coverages):
        if coverage > threshold:
            #print(index)
            new_lines.append(lines[index])
    
    # plot
    plot_lines(x, y, new_lines)

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

def plot_zIdPhi(zIdPhi_values):
    # Collect all zIdPhi values from all trial types
    all_zIdPhis = []
    for zIdPhis in zIdPhi_values.values():
        all_zIdPhis.extend(zIdPhis)
    
    # Convert to a NumPy array for statistical calculations
    all_zIdPhis = np.array(all_zIdPhis)
    
    # Create a single plot
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(all_zIdPhis, bins=30, alpha=0.7, label='All Trial Types')
    
    # Calculate and plot the mean and standard deviation lines
    mean = np.mean(all_zIdPhis)
    std = np.std(all_zIdPhis)
    
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=2, label='+1 STD')
    plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=2, label='-1 STD')
    
    # Set the title and labels
    plt.title('Combined IdPhi Distribution Across All Trial Types')
    plt.xlabel('zIdPhi')
    plt.ylabel('Frequency')
    
    # Show the legend
    plt.legend()
    
    plt.tight_layout()
    plt.show()



# CENTRAL METHODS (traversing) -----------
def scatter_plot(data_structure, ratID, day):
    DLC = data_structure[ratID][day]['DLC_tracking']
    
    # get coordinates
    x, y = filter_dataframe(DLC)
    
    # plot
    create_scatter_plot(x, y)

def occupancy_map(data_structure, ratID, day):
    DLC_df = data_structure[ratID][day]['DLC_tracking']
    SS_df = data_structure[ratID][day]['stateScriptLog']
    
    # get x and y coordinates
    x, y = filter_dataframe(DLC_df)
    
    # get framerate
    framerate = calculate_framerate(SS_df, x)
    
    # make map
    create_occupancy_map(x, y, framerate)

def speed(data_structure, ratID, day):
    DLC_data = data_structure[ratID][day]['DLC_tracking']
    SS_data = data_structure[ratID][day]['stateScriptLog']
    
    track_part = 'haunch' # assumed to be the best for tracking animal speed
    x, y = filter_dataframe(DLC_data, track_part)
    
    # get framerate
    framerate = calculate_framerate(SS_data, x)
    
    # calculate speed
    speed = calculate_speed(x, y, framerate)
    
    return speed

def time_until_first_choice(data_structure, ratID, day):
    content = data_structure[ratID][day]['stateScriptLog']
    
    time = time_until_choice(content)
    print(time)

def quantify_VTE(data_structure, ratID, day, save = False):
    DLC_df = data_structure[ratID][day]['DLC_tracking']
    SS_df = data_structure[ratID][day]['stateScriptLog']
    timestamps = data_structure[ratID][day]['videoTimeStamps']
    
    # check timestamps
    check_timestamps(DLC_df, timestamps)
    
    # get x and y coordinates
    x, y = filter_dataframe(DLC_df)
    
    # define zones
    #home_hull = define_zones(x, y, x_min = 850, x_max = 1050, y_min = 0, y_max = 250, will_plot = True)
    #arm_3_hull = define_zones(x, y, x_min = 150, x_max = 370, y_min = 0, y_max = 250)
    #arm_5_hull = define_zones(x, y, x_min = 150, x_max = 370, y_min = 700, y_max = 930)
    #arm_7_hull = define_zones(x, y, x_min = 850, x_max = 1100, y_min = 700, y_max = 960)
    ellipse_params = get_centre_zone(x, y, ratID, day, save)
    
    # do a time delay before something counts as outside the hull / set upper and lower bound for  Idphi
    # get trial start times + trial type
    trial_starts = get_trial_start_times(timestamps, SS_df)
    
    # calculate IdPhi for each trial
    IdPhi_values = {}
    
    for trial_start, trial_type in trial_starts.items():
        # cut out the trajectory for each trial
        # look through points starting at trial start time to see when it goes into different hulls
        past_inside = False # this checks if any point has ever been inside hull for this iteration of loop
        trajectory_x = []
        trajectory_y = []
        
        trial_start = math.floor(trial_start) # round down so it can be used as an index
        
        for index in range(trial_start, len(x)):
            x_val = x.iloc[index]
            y_val = y.iloc[index]
            
            # skip loop of x or y is NaN
            if math.isnan(x_val) or math.isnan(y_val):
                continue
            
            #point = (x_val, y_val)
            #inside = check_if_inside(point, centre_hull)
            inside = is_point_in_ellipse(x_val, y_val, ellipse_params)
            
            if inside:
                past_inside = True
                trajectory_x.append(x_val)
                trajectory_y.append(y_val)
            else:
                if past_inside:
                    break # ok so now it has exited the centre hull
        
        # calculate Idphi of this trajectory
        IdPhi = calculate_IdPhi(trajectory_x, trajectory_y)
        
        # store IdPhi according to trial type
        if trial_type not in IdPhi_values:
            IdPhi_values[trial_type] = []
        IdPhi_values[trial_type].append(IdPhi)
    
    # calculate zIdPhi according to trial types
    zIdPhi_values = {}
    for trial_type, IdPhis in IdPhi_values.items():
        zIdPhi = zscore(IdPhis)
        zIdPhi_values[trial_type] = zIdPhi
    
    #plot_zIdPhi(zIdPhi_values)
    
    return zIdPhi_values, IdPhi_values

def get_centre_zone(x, y, ratID, day, save = False): #currently highly experimental, ask cat for an exp if needed
    lines = generate_lines(x, y)
    
    # determine whether saving is necessary
    file_path = f'/Users/catpillow/Downloads/VTE_Data/{ratID}/{day}/coverage_scores.csv'
    
    if save or not os.path.exists(file_path): # saves regardless if file doesn't exist
        coverages = calculate_line_coverages(x, y, lines)
        df = pd.DataFrame(coverages, columns=['Coverages'])
        df.to_csv(file_path)
    else:
        df = pd.read_csv(file_path)
        coverages = df['Coverages'].tolist()
    
    # filter out lines that don't pass the threshold
    threshold = 0.4
    updated_lines = make_new_lines(lines, coverages, threshold)
    
    #print(updated_lines)
    #plot_lines(x, y, updated_lines)
    
    # intersection points
    intersections = find_intersections(updated_lines)
    
    # create convex hull
    if intersections:
        points = np.array(intersections)
        points = points[~np.isinf(points).any(axis = 1)]
        unique_points = np.unique(points, axis = 0)
        ellipse_params = make_ellipse(unique_points)
    else:
        print('no intersections')
    
    plot_ellipse(ellipse_params, x, y)
    plot_lines(x, y, lines)
    
    return ellipse_params

def test(data_structure, ratID, day):
    DLC_df = data_structure[ratID][day]['DLC_tracking']
    
    # get x and y coordinates
    x, y = filter_dataframe(DLC_df)
    
    get_centre_zone(x, y)

def rat_VTE_over_sessions(data_structure, ratID):
    rat_path = f'/Users/catpillow/Downloads/VTE_Data/{ratID}'
    
    sum_zIdPhi = []
    
    for day_folder in os.listdir(rat_path): # loop for each day (in each rat folder)
        zIdPhi_values = quantify_VTE(data_structure, ratID, day_folder, save = True)
        
        for zIdPhi in zIdPhi_values:
            sum_zIdPhi.append(zIdPhi)
    
    plot_zIdPhi(sum_zIdPhi)
    
    return
    

# ASSIGNMENT 1 --------
# creating the main data structure
#base_path = '/Users/catpillow/Downloads/Data 2'
#main_data_structure = data_structure.create_main_data_structure(base_path)

# saving
save_path = '/Users/catpillow/Downloads/VTE_Data'
#data_structure.save_data_structure(main_data_structure, save_path)

# loading
loaded_data_structure = data_structure.load_data_structure(save_path)

# ASSIGNMENT 2 ---------
# example
ratID = 'TH405'
day = 'Day7'

# plot positioning for greenLED
#scatter_plot(loaded_data_structure, ratID, day)

# occupancy map
#occupancy_map(loaded_data_structure, ratID, day)

# calculate speed
#speed(loaded_data_structure, ratID, day)

# ASSIGNMENT 3 ---------
performance_analysis.rat_performance_one_session(loaded_data_structure, ratID, day)

# ASSIGNMENT 4 ---------
#time_until_first_choice(loaded_data_structure, ratID, day)

# ASSIGNMENT 5 --------
#zIdPhi, IdPhi = quantify_VTE(loaded_data_structure, ratID, day, True)
#print(f"zIdPhi - {zIdPhi}")
#print(f"IdPhi - {IdPhi}")
#test(loaded_data_structure, ratID, day)

#rat_VTE_over_sessions(loaded_data_structure, ratID)

# LEARNING RATES --------
#rat_performance = performance_analysis.rat_performance_over_sessions(loaded_data_structure, ratID)
performance_analysis.create_all_rats_performance(loaded_data_structure, save_path = save_path)
    
# for loading
'''with open(pickle_path, 'rb') as fp:
    rat_performance = pickle.load(fp)'''
