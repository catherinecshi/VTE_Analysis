"""custom common errors"""

from config import settings

class ExpectationError(Exception):
    """Exception raised when input value is different from expected value"""
    
    def __init__(self, input_value, expected_values):
        self.input_value = input_value
        self.expected_values = expected_values
        self.message = f"Expected {expected_values} but got {input_value} for {settings.CURRENT_RAT} {settings.CURRENT_DAY} {settings.CURRENT_TRIAL}"
        
        super().__init__(self.message)

class UnexpectedNoneError(Exception):
    """Exception raised for none types where there shouldn't be one"""
    
    def __init__(self, function, variable):
        self.function = function
        self.variable = variable
        self.message = f"In {function} {variable} is unexpected None for {settings.CURRENT_RAT} {settings.CURRENT_DAY} {settings.CURRENT_TRIAL}"
        
        super().__init__(self.message)

class LengthMismatchError(Exception):
    """Exception raised for errors where two things should be the same length but are not"""
    
    def __init__(self, first_length, second_length, function):
        self.first_length = first_length
        self.second_length = second_length
        self.function = function
        self.message = f"Mismatch of the two lengths {first_length} vs {second_length} in {function} for {settings.CURRENT_RAT} {settings.CURRENT_DAY}"
        
        super().__init__(self.message)

class CorruptionError(Exception):
    """Exception raised when file cannot be processed"""
    
    def __init__(self, file, function):
        self.file = file
        self.function = function
        self.message = f"error with {settings.CURRENT_RAT} on {settings.CURRENT_DAY}, for file {file} {function}"
        
        super().__init__(self.message)

class NoMatchError(Exception):
    """Exception raised when no match is found when using re"""
    
    def __init__(self, pattern, input_value, function):
        self.pattern = pattern
        self.input_value = input_value
        self.function = function
        self.message = f"error with {settings.CURRENT_RAT} on {settings.CURRENT_DAY} where no match with {pattern} was found in {input_value} within {function}"
        
        super().__init__(self.message)

class NoPathError(Exception):
    """Exception raised when no path is found where there should be"""
    
    def __init__(self, path, function):
        self.path = path
        self.function = function
        self.message = f"error with {settings.CURRENT_RAT} on {settings.CURRENT_DAY} where {path} cannot be found in {function}"
        
        super().__init__(self.message)