# packages
import os
import bisect
import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import zscore

# other helper files
import data_structure
import performance_analysis

# DATA MANIPULATION/ANALYSIS METHODS ---------
def filter_dataframe(df, track_part = 'greenLED', std_multiplier = 7, eps = 70, min_samples = 40, distance_threshold = 190, start_index = None): # currently keeps original indices
    # modify a copy instead of the original
    # also filter based on likelihood values
    likely_data = df[df[(track_part, 'likelihood')] > 0.999].copy()
    
    # filter out points before the rat has started its first trial
    if start_index:
        likely_data = likely_data[likely_data.index >= start_index]
    
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
    #print(timestamps)
    
    for index, time in enumerate(timestamps): #loops through timestamps
        if index == 0:
            continue
    
        # calculate diff in seconds between frames
        current_time = timestamps[index]
        past_time = timestamps[index - 1]
        time_diff = current_time - past_time
        
        # make sure it's around the 0.03 range
        if time_diff > 0.05 or time_diff < 0.01:
            #print(time_diff)
            time_off[index] = time_diff # time is off here
            
            if index_off < 5:
                #print(f"time_diff is off norm for {index}")
                index_off += 1
        else:
            continue
    
    # second check - make sure x and timestamps are the same length
    if not(len(df) == len(timestamps)):
        '''print("length of x and timestamps don't match up")
        print(len(df))
        print(len(timestamps))'''
        
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
        
        elif not line[0].isdigit(): # check if the first char is a number - skip if not
            # hopefully this takes cares of the weird errors wth pressng summary after new trial showed
            continue
        
        elif start_of_trial and "trialType" in line: # store trial type
            parts = line.split()
            trial_type = parts[3]
            trial_info[trial_starts[-1]] = trial_type # assumes this will always come after New Trial'
            start_of_trial = False
                  
        elif 'New Trial' in line: # indicate start of a new trial
            start_of_trial = True
            
            # store the time during this event
            parts = line.split()
            trial_start = parts[0]
            trial_info[trial_start] = None
            trial_starts.append(trial_start)
    
    video_starts = video_trial_starts(timestamps, trial_starts) # this should be the indices for x/y where trials start
    
    # change trial_info such that the key is video_starts instead of trial_starts
    video_trial_info = {}
    
    if len(video_starts) == len(trial_starts):
        for index, video_start in enumerate(video_starts):
            original_start_time = trial_starts[index]
            trial_type = trial_info.get(original_start_time) # get trial type for trial start time
            video_trial_info[video_start] = trial_type
    
    return video_trial_info

def video_trial_starts(timestamps, SS_times):
    trial_starts = []
    
    for time in SS_times:
        # ensure consistent data types
        time = float(int(time) / 1000)
        
        if time in timestamps: # if there is a perfect match between MCU & ECU and the time is in timestamps
            index = timestamps.index(time)
            trial_starts.append(index)
        else: # if there isn't a perfect match
            #print(f"Imperfect match between ECU and MCU at {time}")
            
            # index where time is inserted into timestamps
            idx = bisect.bisect_left(timestamps, time)
            
            # check neighbours for closest time
            if idx == 0: # if time < any available timestamp, so = 0
                trial_starts.append(0)
            elif idx == len(timestamps): # if time > any available timestamp, so = len(timestamps)
                trial_starts.append(len(timestamps))
            else:
                before = timestamps[idx - 1]
                after = timestamps[idx]
                closest_time = before if (time - before) <= (after - time) else after
                index = np.where(timestamps == closest_time)[0][0]
                trial_starts.append(index)
    
    return trial_starts # with this, each trial_start is the index of the time when trial starts in relation to timestamps

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
    #print(values)
    
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
                #print("window_len > 1")
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
                        #print("model too far from actual results")
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
    #print("got derivative for x")
    dy = derivative(trajectory_y, sr, d, m)
    #print("got derivative for y")
    
    # calculate + unwrap angular velocity
    Phi = np.arctan2(dy, dx)
    Phi = np.unwrap(Phi)
    dPhi = derivative(Phi, sr, d, m)
    #print("got derivative for Phi")
    
    # integrate change in angular velocity
    IdPhi = np.trapz(np.abs(dPhi))
    
    #print("returning IdPhi")
    return IdPhi

#def learning_rates_vs_VTEs():
    

            
# GETTING ZONES -----------
def generate_lines(x, y, gap_between_lines = 20, degree_step = 10, min_length = 950, hv_line_multiplier = 2):
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
            
    return lines

def calculate_line_coverages(x, y, lines, num_segments = 15, threshold = 5):
    """calculates how well each line is covered by points based on how many line segments have points in its vicinity

    Args:
        x (np array): x values
        y (np array): y values
        lines (_type_): list including slope and intercept values, making up y = mx + b
        num_segments (int, optional): the number of segments for testing lines' fit. Defaults to 15.
        threshold (int, optional): distance a point can be from a line before it's not considered in its vicinity. Defaults to 5.

    Returns:
        coverage_scores (np int array): how many segments on the line are covered by points
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
    
    '''# plot distance to know what's a good threshold
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
    plt.show()'''
    
    return coverage_scores, starts, ends

def make_new_lines(lines, coverages, starts, ends, threshold):
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
    x, y = point
    
    if vertical:
        point_inside = start <= y <= end
    else:
        point_inside = start <= x <= end
    
    return point_inside

def find_intersections(lines, starts, ends):
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

def make_convex_hull(intersection_points, x, y, day = None):
    #Perform DBSCAN clustering
    dbscan = DBSCAN(eps=10, min_samples=5)  # Adjust these parameters as necessary
    clusters = dbscan.fit_predict(intersection_points)

    # Find the cluster with the most points (highest concentration)
    cluster_indices, counts = np.unique(clusters, return_counts=True)
    densest_cluster_index = cluster_indices[np.argmax(counts)]
    densest_cluster_points = intersection_points[clusters == densest_cluster_index]

    # Create a convex hull around the densest cluster
    hull = ConvexHull(densest_cluster_points)
    
    plt.scatter(x, y)

    # Plotting (optional, for visualization)
    plt.scatter(intersection_points[:,0], intersection_points[:,1], alpha=0.5, color = 'green')
    plt.scatter(densest_cluster_points[:,0], densest_cluster_points[:,1], color='red')
    for simplex in hull.simplices:
        plt.plot(densest_cluster_points[simplex, 0], densest_cluster_points[simplex, 1], 'k-')

    # Create a Polygon patch for the convex hull
    hull_polygon = Polygon(densest_cluster_points[hull.vertices], closed=True, edgecolor='k', fill=False)
    plt.gca().add_patch(hull_polygon)
    #plt.show()
    
    if day:
        plt.savefig(f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13/{day}')
    
    return hull, densest_cluster_points


# PLOTTING METHODS --------
def create_scatter_plot(x, y):
    plt.figure(figsize = (10, 6))
    #plt.scatter(x, y, c = 'green', alpha = 0.6)
    plt.scatter(x, y, color='green', alpha=0.4)
    plt.title('VTEs vs Learning Rate')
    plt.xlabel('Number of VTEs in a Session')
    plt.ylabel('Change in Performance')
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
    
    # rotating so it looks like scatter plot
    rotated_occupancy_grid = np.rot90(occupancy_grid)
    
    # plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(rotated_occupancy_grid, cmap=custom_cmap, interpolation='nearest')
    plt.colorbar(label='Time spent in seconds')
    plt.title('Occupancy Map')
    plt.xlabel('X Bins')
    plt.ylabel('Y Bins')
    plt.show()

def plot_animation(x, y, trajectory_x = None, trajectory_y = None, interval = 20, highest = 0, zIdPhi = None):
    if not trajectory_x: # this is for when you want to plot the entire trajectory throughout the trial
        trajectory_x = x
    
    if not trajectory_y:
        trajectory_y = y
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha = 0.2) # plot the totality first
    line, = ax.plot([], [], 'bo-', linewidth = 2) # line plot
    
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
    
    # set title accordingly
    if highest == 1: # is highest
        plt.title('VTE Trial')
    elif highest == 2: # is lowest
        plt.title('Non-VTE Trial')
    
    #plt.show()
    ani.save('/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13')
    
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

def plot_zIdPhi(zIdPhi_values, day = None):
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
    #plt.show()
    if day:
        plt.savefig(f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13/{day}')

def plot_trajectory(x, y, zIdPhi, trajectories, highest):
    
    # get trajectory points
    trajectory_x, trajectory_y = trajectories
    
    # plot the normal points
    plt.figure(figsize = (10, 6))
    plt.plot(x, y, color='green', alpha=0.4)
    
    # plot the trajectory
    plt.plot(trajectory_x, trajectory_y, color = 'red', alpha = 0.8, label = zIdPhi)
    
    # display plot
    if highest:
        plt.title('VTE Trial trajectory')
    else:
        plt.title('Non-VTE Trial Trajectory')
        
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_segments(x, y, lines, starts, ends, day = None):
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
    plt.grid(True)
    #plt.show()
    
    # save the figure
    if day:
        plt.savefig(f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13/{day}')


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
    """DLC_df = data_structure[ratID][day]['DLC_tracking']
    SS_df = data_structure[ratID][day]['stateScriptLog']
    timestamps = data_structure[ratID][day]['videoTimeStamps']"""
    
    DLC_df = data_structure[day]['DLC_tracking']
    SS_df = data_structure[day]['stateScriptLog']
    timestamps = data_structure[day]['videoTimeStamps']
    
    # check timestamps
    check_timestamps(DLC_df, timestamps)
    
    # do a time delay before something counts as outside the hull / set upper and lower bound for  Idphi
    # get trial start times + trial type
    trial_starts = get_trial_start_times(timestamps, SS_df)
    
    # get x and y coordinates
    # get the first trial start time to pass into filtering
    first_trial_start = next(iter(trial_starts))
    x, y = filter_dataframe(DLC_df, start_index=first_trial_start)
    
    # define zones
    #home_hull = define_zones(x, y, x_min = 850, x_max = 1050, y_min = 0, y_max = 250, will_plot = True)
    #arm_3_hull = define_zones(x, y, x_min = 150, x_max = 370, y_min = 0, y_max = 250)
    #arm_5_hull = define_zones(x, y, x_min = 150, x_max = 370, y_min = 700, y_max = 930)
    #arm_7_hull = define_zones(x, y, x_min = 850, x_max = 1100, y_min = 700, y_max = 960)
    centre_hull = get_centre_zone(x, y, ratID, day, save)
    
    # calculate IdPhi for each trial
    IdPhi_values = {}
    
    # store trajectories for plotting later
    trajectories = {}
    
    for trial_start, trial_type in trial_starts.items(): # where trial type is a string of a number corresponding to trial type
        # cut out the trajectory for each trial
        # look through points starting at trial start time to see when it goes into different hulls
        past_inside = False # this checks if any point has ever been inside hull for this iteration of loop
        trajectory_x = []
        trajectory_y = []
        
        trial_start = math.floor(trial_start) # round down so it can be used as an index
        
        for index in range(trial_start, len(timestamps)): # x has been filtered so is not an appropriate length now
            # getting x and y
            if index == trial_start:
                print(index)
            
            if index in x.index and index in y.index:
                x_val = x.loc[index] # loc is based on actual index, iloc is based on position
                y_val = y.loc[index]
            elif index == trial_start:
                print(f'trial started and cannot find x and y values - {trial_start}')
                continue
            else:
                continue
            
            """point = (x_val, y_val)
            inside = check_if_inside(point, centre_hull)
            
            if inside:
                past_inside = True
                trajectory_x.append(x_val)
                trajectory_y.append(y_val)
            else:
                if past_inside:
                    break # ok so now it has exited the centre hull"""
            
            if index < trial_start + 1000:
                trajectory_x.append(x_val)
                trajectory_y.append(y_val)
            else:
                break
        
        # calculate Idphi of this trajectory
        IdPhi = calculate_IdPhi(trajectory_x, trajectory_y)
        #plot_animation(x, y, trajectory_x = trajectory_x, trajectory_y= trajectory_y)
        
        # store IdPhi according to trial type
        if trial_type not in IdPhi_values:
            IdPhi_values[trial_type] = []
        IdPhi_values[trial_type].append(IdPhi)
        
        # store each trajectory for plotting latter
        if trial_type in trajectories:
            trajectories[trial_type].append((trajectory_x, trajectory_y))
        else:
            trajectories[trial_type] = [(trajectory_x, trajectory_y)]
    
    # calculate zIdPhi according to trial types
    zIdPhi_values = {}
    highest_zIdPhi = None
    highest_trajectories = None
    lowest_zIdPhi = None
    lowest_trajectories = None
    
    # this z scores according to trial type
    for trial_type, IdPhis in IdPhi_values.items():
        zIdPhis = zscore(IdPhis)
        zIdPhi_values[trial_type] = zIdPhis
        
        for i, zIdPhi in enumerate(zIdPhis): # this is to get the highest and lowest zidphi for plotting vte/non
            if highest_zIdPhi:
                if zIdPhi > highest_zIdPhi:
                    highest_zIdPhi = zIdPhi
                    highest_trajectories = trajectories[trial_type][i]
            else:
                highest_zIdPhi = zIdPhi
                highest_trajectories = trajectories[trial_type][i]
            
            if lowest_zIdPhi:
                if zIdPhi < lowest_zIdPhi and len(trajectories[trial_type][i]) > 2:
                    lowest_zIdPhi = zIdPhi
                    lowest_trajectories = trajectories[trial_type][i]
            else:
                lowest_zIdPhi = zIdPhi
                lowest_trajectories = trajectories[trial_type][i]
    
    highest_trajectory_x, highest_trajectory_y = highest_trajectories
    lowest_trajectory_x, lowest_trajectory_y = lowest_trajectories
    
    plot_zIdPhi(zIdPhi_values)
    plot_animation(x, y, highest_trajectory_x, highest_trajectory_y, highest = 2, zIdPhi=highest_zIdPhi)
    plot_animation(x, y, lowest_trajectory_x, lowest_trajectory_y, highest = 1, zIdPhi=lowest_zIdPhi)
    
    return zIdPhi_values, IdPhi_values, trajectories

def get_centre_zone(x, y, ratID, day, save = False, threshold = None): #currently highly experimental, ask cat for an exp if needed
    # determine whether saving is necessary
    file_path = f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/{ratID}/{day}'
    coverage_path = file_path + '/coverage_scores.csv'
    covered_path = file_path + '/covered_lines.csv'
    intersections_path = file_path + '/intersections.csv'
    hull_path = file_path + '/hull_vertices.npy'
    
    if save or not os.path.exists(coverage_path) or not os.path.exists(hull_path): # saves regardless if file doesn't exist
        lines = generate_lines(x, y)
        coverages, starts, ends = calculate_line_coverages(x, y, lines)
        
        # filter out lines that don't pass the threshold
        updated_lines, updated_starts, updated_ends = make_new_lines(lines, coverages, starts, ends, threshold)
        
        #plot_lines(x, y, lines)
        plot_segments(x, y, updated_lines, updated_starts, updated_ends, day = day)
        #print(updated_lines)
        
        # intersection points
        intersections = find_intersections(updated_lines, updated_starts, updated_ends)
        intersection_points = np.array(intersections) # np array for DBSCAN to work
        
        # create convex hull
        hull, densest_cluster_points = make_convex_hull(intersection_points, x, y, day = day)
        
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

def test(data_structure, ratID, day):
    DLC_df = data_structure[ratID][day]['DLC_tracking']
    
    # get x and y coordinates
    x, y = filter_dataframe(DLC_df)
    
    get_centre_zone(x, y)

def rat_VTE_over_sessions(data_structure, ratID):
    rat_path = f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/{ratID}'
    
    for day in data_structure:
        try:
            zIdPhi, IdPhi, trajectories = quantify_VTE(data_structure, ratID, day, save = False)
            zIdPhi_path = os.path.join(rat_path, day, 'zIdPhi.npy')
            IdPhi_path = os.path.join(rat_path, day, 'IdPhi.npy')
            trajectories_path = os.path.join(rat_path, day, 'trajectories.npy')
            # save 
            with open(zIdPhi_path, 'wb') as fp:
                pickle.dump(zIdPhi, fp)
            
            with open(IdPhi_path, 'wb') as fp:
                pickle.dump(IdPhi, fp)
            
            with open(trajectories_path, 'wb') as fp:
                pickle.dump(trajectories, fp)
        except Exception as error:
            print(f'error - {error} on day {day}')
            


### VTE ANALYSIS --------
def compare_zIdPhis(base_path):
    IdPhis_across_days = [] # this is so it can be zscored altogether
    
    days = []
    zIdPhi_means = []
    zIdPhi_stds = []
    IdPhi_means = []
    IdPhi_stds = []
    vte_trials = []
    
    for day_folder in os.listdir(base_path):
        day_path = os.path.join(base_path, day_folder)
        if os.path.isdir(day_path):
            days.append(day_folder)
        
        for root, dirs, files in os.walk(day_path):
            for f in files:
                file_path = os.path.join(root, f)
                if 'zIdPhi' in f:
                    with open(file_path, 'rb') as fp:
                        zIdPhis = pickle.load(fp)
                    
                    # flatten from dict to array
                    all_zIdPhis = [zIdPhi for zIdPhi_vals in zIdPhis.values() for zIdPhi in zIdPhi_vals]
                    
                    # mean & std stored across days
                    zIdPhis_mean = np.mean(all_zIdPhis)
                    zIdPhis_std = np.std(all_zIdPhis)
                    
                    # append into array
                    zIdPhi_means.append(zIdPhis_mean)
                    zIdPhi_stds.append(zIdPhis_std)
                    
                    # check how many vte trials
                    cutoff = zIdPhis_mean + zIdPhis_std
                    vte_trials.append(sum(zIdPhi > cutoff for zIdPhi in all_zIdPhis))
                
                if 'IdPhi' in f and 'z' not in f:
                    with open(file_path, 'rb') as fp:
                        IdPhis = pickle.load(fp)
                    
                    # flatten
                    all_IdPhis = [IdPhi for IdPhi_vals in IdPhis.values() for IdPhi in IdPhi_vals]
                
                    IdPhis_mean = np.mean(all_IdPhis)
                    IdPhis_std = np.std(all_IdPhis)
                    
                    IdPhi_means.append(IdPhis_mean)
                    IdPhi_stds.append(IdPhis_std)
                    
                    # zscore later perhaps
                    # IdPhis_across_days.append(all_IdPhis)
    
    """print(len(zIdPhi_means))
    print(len(vte_trials))
    print(len(IdPhi_means))
    print(len(days))"""
    
    df = pd.DataFrame({
        'Day': day,
        'zIdPhi Mean': zIdPhi_means,
        'zIdPhi Std': zIdPhi_stds,
        'IdPhi Mean': IdPhi_means,
        'IdPhi Std': IdPhi_stds,
        'VTE Trials': vte_trials
    })
    
    # sort according to day number
    df['sort_key'] = df['Day'].apply(lambda x: int(x[3:])) 
    df_sorted = df.sort_values(by = 'sort_key')
    df_sorted = df_sorted.drop(columns = ['sort_key']) # drop now that it's sorted
    
    """# calculate difference between values for consecutive days
    comparison_cols = df.columns.drop('Day')
    diffs = df[comparison_cols].diff()
    # diffs['Day'] = df['Day'] # add days back in if desired"""
    
    # save dataframe
    dataframe_path = os.path.join(base_path, 'zIdPhis_and_IdPhis')
    df.to_csv(dataframe_path)
    
    """# save differences in a separate numpy file
    IdPhi_mean_diffs = diffs['IdPhi Mean'].to_numpy()
    zIdPhi_mean_diffs = diffs['zIdPhi Mean'].to_numpy()
    
    idphi_diffs_path = os.path.join(base_path, 'IdPhi_Mean_Diffs')
    zidphi_diffs_path = os.path.join(base_path, 'zIdPhi_Mean_Diffs')
    
    np.save(idphi_diffs_path, IdPhi_mean_diffs)
    np.save(zidphi_diffs_path, zIdPhi_mean_diffs)
    
    print(f'idphi - {IdPhi_mean_diffs}')
    print(f'zidphi - {zIdPhi_mean_diffs}')"""
    
    # return sorted vte trials according to day
    vte_trials_sorted = df_sorted['VTE Trials']
    
    return vte_trials_sorted
    

# ASSIGNMENT 1 --------
# creating the main data structure
#base_path = '/Users/catpillow/Documents/VTE Analysis/Data_draft'
#main_data_structure = data_structure.create_main_data_structure(base_path)

# saving
#base_path = '/Users/catpillow/Downloads/BP13_timestamps'
#save_path = '/Users/catpillow/Documents/VTE Analysis/VTE_Data' # this is just SS (added BP13 DLC & timestamps)
save_path_BP = '/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13'
#save_path = '/Users/catpillow/Downloads/VTE_Data'
#data_structure.save_data_structure(main_data_structure, save_path)
#data_structure.save_DLC(base_path, save_path)
#data_structure.save_timestamps(base_path, save_path)

# loading
#loaded_data_structure = data_structure.load_data_structure(save_path)
BP13_data = data_structure.load_one_rat(save_path_BP)

# ASSIGNMENT 2 ---------
# example
ratID = 'BP13'
day = 'Day8'

# plot positioning for greenLED
#scatter_plot(loaded_data_structure, ratID, day)

# occupancy map
#occupancy_map(loaded_data_structure, ratID, day)

# calculate speed
#speed(loaded_data_structure, ratID, day)

# ASSIGNMENT 3 ---------
#performance_analysis.rat_performance_one_session(loaded_data_structure, ratID, day)

# ASSIGNMENT 4 ---------
#time_until_first_choice(loaded_data_structure, ratID, day)

# VTEs --------
#DLC = loaded_data_structure[ratID][day]['DLC_tracking']
    
# get coordinates
#x, y = filter_dataframe(DLC)

#plot_animation(x, y)
zIdPhi, IdPhi, trajectories = quantify_VTE(BP13_data, ratID, day, save = False)
#rat_VTE_over_sessions(BP13_data, ratID)
#print(f"zIdPhi - {zIdPhi}")
#print(f"IdPhi - {IdPhi}")
#test(loaded_data_structure, ratID, day)

#rat_VTE_over_sessions(loaded_data_structure, ratID)

# LEARNING RATES --------
#rat_performance = performance_analysis.rat_performance_over_sessions(loaded_data_structure, ratID)
#performance_analysis.create_all_rats_performance(loaded_data_structure, save_path = save_path)
#all_rats_performances = performance_analysis.load_rat_performance(save_path)
#performance_analysis.plot_all_rat_performances(all_rats_performances)
#perf_changes = None
"""for rat, rat_performance in all_rats_performances.items():
    if rat == ratID:
        performance_analysis.plot_rat_perf_changes(rat_performance)"""
        #perf_changes, avg_changes = performance_analysis.change_in_performance(rat_performance)
#performance_analysis.all_rats_perf_changes(all_rats_performances)

#performance_analysis.days_until_criteria(all_rats_performances)
#performance_analysis.perf_until_critera(all_rats_performances)


# COMPARISON ---------------
"""vte_trials = compare_zIdPhis(save_path_BP)

print(len(vte_trials))
print(len(avg_changes))
create_scatter_plot(vte_trials, avg_changes)"""