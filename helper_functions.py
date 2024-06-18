import numpy as np
from scipy.spatial import Delaunay

### GET VALUE --------------
def get_time(statescript_time):
    # converts statescript time to seconds since start of recording
    time_passed = statescript_time / 1000 # turning it from ms to seconds
    
    return time_passed

def get_framerate():
    # returns framerate, should usually be the same, or check_timestamps will catch it
    framerate = 0.03 # seconds
    
    return framerate

def calculate_speed(x, y, framerate):
    diff_x = x.diff().fillna(0)
    diff_y = y.diff().fillna(0)
    
    displacement_per_frame = np.sqrt(diff_x**2 + diff_y**2)
    
    speed = displacement_per_frame * framerate
    
    return speed

def get_speed_session(data_structure, ratID, day):
    # gets the average speed of the rat for the entire session
    DLC_data = data_structure[ratID][day]['DLC_tracking']
    SS_data = data_structure[ratID][day]['stateScriptLog']
    
    track_part = 'haunch' # assumed to be the best for tracking animal speed
    x, y = data_structure.filter_dataframe(DLC_data, track_part)
    
    # get framerate
    framerate = get_framerate(SS_data, x)
    
    # calculate speed
    speed = calculate_speed(x, y, framerate)
    
    return speed

def get_time_until_choice(data_structure, ratID, day):
    content = data_structure[ratID][day]['stateScriptLog']
    
    # calculates the time the rat takes until its first choice in the session
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


### CHECKING THINGS ARE NORMAL ------------
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
            time_off[index] = time_diff # time is off here
            
            if index_off < 5:
                index_off += 1
        else:
            continue
    
    # second check - make sure x and timestamps are the same length
    if not(len(df) == len(timestamps)):
        diff = len(df) - len(timestamps)
        
        # it seems like most of them differ by 1, where df = timestamps - 1, so i'm doing a rough subtraction here
        if diff == 1:
            timestamps.pop()
    
    return timestamps

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

def is_point_in_hull(point, hull):
    # delaunay triangulation of hull points
    del_tri = Delaunay(hull.points[hull.vertices])
    
    # check if point is inside the hull
    return del_tri.find_simplex(point) >= 0