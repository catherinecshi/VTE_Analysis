"""
useful hard-coded conversions
"""
from typing import Union

from config import settings
from debugging import error_types

# ==============================================================================
# TRIAL TYPE CONVERSIONS
# ==============================================================================

def string_to_int_trial_types(string_trial: str) -> int:
    """
    converts string trial type (like "AB") to integer
    raises error if string_trial isn't a possible trial type
    """
    # Use EF mappings if current rat is in RATS_WITH_EF, otherwise use regular mappings
    if settings.CURRENT_RAT in settings.RATS_WITH_EF:
        mappings = settings.TRIAL_TYPE_MAPPINGS_EF
    else:
        mappings = settings.TRIAL_TYPE_MAPPINGS
    
    int_trial_type = mappings.get(string_trial)
    
    if int_trial_type is None:
        raise error_types.ExpectationError(string_trial, mappings)
    else:
        return int_trial_type

def int_to_string_trial_types(int_trial: int) -> str:
    """
    converts int trial type to string (like "AB")
    raises error if int_trial isn't a possible tiral type
    """
    # Use EF mappings if current rat is in RATS_WITH_EF, otherwise use regular mappings
    if settings.CURRENT_RAT in settings.RATS_WITH_EF:
        mappings = settings.TRIAL_TYPE_MAPPINGS_EF
    else:
        mappings = settings.TRIAL_TYPE_MAPPINGS
    
    reverse_mapping = {v: k for k, v in mappings.items()}
    
    string_trial_type = reverse_mapping.get(int_trial)
    if string_trial_type is None:
        raise error_types.ExpectationError(int_trial, reverse_mapping)
    else:
        return string_trial_type

# ==============================================================================
# TRIAL TYPE CORRECTNESS
# ==============================================================================

def choice_to_correctness(trial_type: int, choice: str) -> bool:
    """
    takes trial type and choice and returns whether it was a correct trial
    does not take string trial type
    
    raises error if choice string cannot be found in trial_type
    """
    string_trial_type = int_to_string_trial_types(trial_type)
    
    if choice == string_trial_type[0]:
        return True
    elif choice == string_trial_type[1]:
        return False
    else:
        raise error_types.ExpectationError(choice, string_trial_type)

def get_other_element(first_element: str, is_correct: bool) -> str:
    """
    takes one element and correct/incorrect and returns the other element
    assumes training pairs and not testing pairs
    """
    
    alphabet = "ABCDEF"
    
    if first_element in alphabet:
        current_index = alphabet.index(first_element)
    else:
        raise error_types.ExpectationError(first_element, alphabet)
    
    if is_correct:
        if current_index == len(alphabet) - 1: # if F
            raise error_types.ExpectationError(first_element, is_correct)
        return alphabet[current_index + 1] # the "next" alphabet letter
    else:
        if current_index == 0: # if A
            raise error_types.ExpectationError(first_element, alphabet)
        return alphabet[current_index - 1] # the "previous" alphabet letter

def type_to_choice(trial_type: Union[int, str], correct: bool) -> str:
    """
    gets the arm the rat went down when given trial_type and whether the rat got the choice correct

    Parameters:
    - trial_type: the number corresponding to the trial type, as shown at the start of statescript comments
    - correct: bool for whether correct

    Returns:
    - str: the letter corresponding to which arm the rat went down
    """
    
    if isinstance(trial_type, str):
        trial_type = int(trial_type)
        
    string_trial_type = int_to_string_trial_types(trial_type)
    
    if correct:
        return string_trial_type[0] # first letter of trial type is correct
    else:
        return string_trial_type[1]

# ==============================================================================
# ELEMENT CONVERSIONS
# ==============================================================================

def letter_to_indices(letter: str) -> int:
    """converts letters representing arms to the index (int) in hierarchy"""
    index = settings.HIERARCHY_MAPPINGS.get(letter)
    
    if index is None:
        raise error_types.ExpectationError(letter, settings.HIERARCHY_MAPPINGS)
    else:
        return index

def type_to_elements(trial_type: Union[int, str], correct: bool) -> tuple[str, str]:
    """
    takes the trial type and correct and then returns the chosen and unchosen element
    
    Parameters:
    - trial_type: the number corresponding to the trial type, as shown at the start of statescript comments
    - correct: bool for whether correct

    Returns:
    - (str): chosen element
    - (str): unchosen element
    """
    if isinstance(trial_type, str):
        trial_type = int(trial_type)
        
    string_trial_type = int_to_string_trial_types(trial_type)
    
    if correct:
        return string_trial_type[0], string_trial_type[1] # first letter of trial type is correct
    else:
        return string_trial_type[1], string_trial_type[0]
    
# ==============================================================================
# CM & PIXEL CONVERSIONS
# ==============================================================================

def convert_pixels_to_cm(trajectory_x: list, trajectory_y: list) -> tuple[list[float], list[float]]:
    """convert list of pixels to list of cm"""
    trajectory_x_cm = [x / 5 for x in trajectory_x]
    trajectory_y_cm = [y / 5 for y in trajectory_y]
    
    return trajectory_x_cm, trajectory_y_cm