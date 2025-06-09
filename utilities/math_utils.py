"""
Mathematical operations
"""

import numpy as np
from typing import Union
from itertools import combinations

from config import settings

def check_equal_length(a, b) -> bool:
    """
    checks if two arrays/dictionaries are the same length (same number of elements or key-value pairs)

    Parameters:
    - a (array or dict): first thing being compared
    - b (array or dict): second thing being compared
        
    Returns:
    - bool: true if the two things are the same length
    """
    
    # have some lenience if it's just like one off
    same_len = -2 < (len(a) - len(b)) < 2

    return same_len

def check_difference(values: list, threshold: Union[int, float]) -> bool:
    """
    check if values in a list are within a certain range of each other

    Parameters:
    - values: list of values with the dfferences to check
    - threshold: acceptable range for values within each other

    Returns:
    - bool: returns True if there are points that are more than threshold away from each other
    """
    
    for a, b in combinations(values, 2):
        if abs(a - b) > threshold:
            return True

    return False # if none are found

def round_to_sig_figs(num: float, sig_figs: int = 3):
    """
    Round a number to specified number of significant figures
    """
    
    if np.isnan(num):
        return num
    if num == 0:
        return 0
    
    return round(num, sig_figs)

def get_sem(successes, totals) -> np.ndarray:
    """
    gets standard error of mean
    does not work with 0 or negative values
    
    Parameters:
    - successes (array-like): success/positive counts
    - totals (array-like): total trial counts (must be > 0)
    
    Returns:
    - numpy.ndarray: Array of standard errors
    """
    if np.any(successes < 0) or np.any(totals < 0):
        raise ValueError(f"for {settings.CURRENT_RAT} {settings.CURRENT_DAY} - successes and totals must be non-negative")
    
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    
    # Direct calculation without extensive error checking
    proportions = successes / totals
    sem_values = np.sqrt(proportions * (1 - proportions) / totals)
    
    return sem_values

def calculate_IdPhi(trajectory_x: Union[np.ndarray, list], trajectory_y: Union[np.ndarray, list], sr: float = settings.FRAMERATE) -> float:
    """
    calculates the integrated angular head velocity (IdPhi) value

    Parameters:
    - trajectory_x: x values for trajectory
    - trajectory_y: y values for trajectory
    - sr: sampling rate. assumes 0.03 (found in settings)

    Returns:
    - float: IdPhi value (singular!!!)
    """
    if isinstance(trajectory_x, list):
        trajectory_x = np.array(trajectory_x)
    
    if isinstance(trajectory_y, list):
        trajectory_y = np.array(trajectory_y)
    
    # derivatives - estimates velocity for each point in time in trajectory
    dx = calculate_derivative(trajectory_x, sr)
    dy = calculate_derivative(trajectory_y, sr)
    
    # triangulate the change in x and y together
    Phi = np.arctan2(dy, dx)
    Phi = np.unwrap(Phi)
    dPhi = calculate_derivative(Phi, sr)
    
    # integrate change in angular velocity sum for each trajectory
    IdPhi = sum(np.abs(dPhi))
    
    return IdPhi

def calculate_derivative(xD: np.ndarray, dT: float, window: float = 1, post_smoothing: float = 0.5, display: bool = False) -> np.ndarray:
    """
    calculates derivate/velocity using adaptive window method
    translated from sj_dxdt in citadel
    
    Parameters:
    - xD: Position vector
    - dT: Time step
    - window: Window size in seconds.
    - postSmoothing: Smoothing window in seconds (0 means no smoothing)
    - display: Whether to print progress
    
    Returns:
    - np.ndarray: Estimated velocity (dx/dt) of position vector xD
    """
    
    # Calculate maximum window size in terms of steps
    nW = min(int(np.ceil(window / dT)), len(xD)) # creates smaller windows if traj is esp long
    nX = len(xD)
    
    # Initialize MSE and slope (b) matrices
    mse = np.zeros((nX, nW)) # MSE approximates how well a straight line fits onto the data
    mse[:, :2] = np.inf
    b = np.zeros((nX, nW)) # this is the same b as y = bx + c
    
    # nan vector for padding
    nanvector = np.full(nW, np.nan)
    
    # Loop over window sizes from 3 to nW
    for iN in range(2, nW):
        if display:
            print('.', end='')
        
        # Calculate slope (b) for current window size iN
        b[:, iN] = np.concatenate((nanvector[:iN], xD[:-iN])) - xD
        b[:, iN] /= iN
        
        # Calculate MSE for the current window size iN
        for iK in range(1, iN + 1):
            q = np.concatenate((nanvector[:iK], xD[:-iK])) - xD + b[:, iN] * iK
            mse[:, iN] += q ** 2
        
        # Average the MSE for each window size
        mse[:, iN] /= iN
    
    if display:
        print('!')

    # Select the window with the smallest MSE for each point - best fit line
    nSelect = np.nanargmin(mse, axis=1)
    dx = np.full_like(xD, np.nan, dtype=float)
    
    # Calculate dx for each point using the optimal window size
    for iX in range(nX):
        dx[iX] = -b[iX, nSelect[iX]] / dT 
    
    # Apply post-smoothing if specified
    if post_smoothing > 0:
        nS = int(np.ceil(post_smoothing / dT))
        dx = np.convolve(dx, np.ones(nS) / nS, mode='same')
    
    return dx