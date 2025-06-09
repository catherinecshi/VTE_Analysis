"""
Central configuration file
Central paths, constants, and settings
"""

# pylint: disable=global-statement

# ==============================================================================
# CONSTANTS
# ==============================================================================

DLC = "DLC_tracking"
SS = "stateScriptLog"
TIMESTAMPS = "videoTimeStamps"
FRAMERATE = 0.03

IMPLANTED_RATS = [
    "BP06", "BP07", "BP12", "BP13", "TH405", "TH508", "BP20", "TH510", "TH605", "TH608"
]

# this is only for numbers found in statescript
TRIAL_TYPE_MAPPINGS = {
    "AB": 1, "BC": 2, "CD": 3, "DE": 4, "EF": 5,
    "BD": 6, "CE": 7, "BE": 8, "AC": 9, "DF": 10
}

# numbers represent index in hierarchy
HIERARCHY_MAPPINGS = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5
}

# ==============================================================================
# DEBUGGING VARIABLES
# ==============================================================================

CURRENT_RAT = ""
CURRENT_DAY = ""
CURRENT_TRIAL = ""

def update_rat(rat: str):
    """use when iterating over rats or days, and debug messages need to know rat & days"""
    global CURRENT_RAT
    CURRENT_RAT = rat

def update_day(day: str):
    """use when iterating over rats or days, and debug messages need to know rat & days"""
    global CURRENT_DAY
    CURRENT_DAY = day

def update_trial(trial: str):
    global CURRENT_TRIAL
    CURRENT_TRIAL = trial
