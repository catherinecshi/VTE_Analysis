"""
logging utilities that allow for hierarchical (script/module) logger creation

USAGE =================
main scripts:
logger = setup_script_logger()  # Auto-detects script name
OR
logger = setup_script_logger("initial_processing")

In modules:
logger = setup_child_logger("data_processing")  # Auto-detects parent script
"""
import os
import sys
import logging
from typing import Optional
from datetime import datetime
from config.paths import paths

def setup_logger(module_name: str, log_level = logging.DEBUG, force_file_handler: bool = False):
    """
    sets up logger with handler
    checks for parent/child relationships and handles accordingly
    """
    logger = logging.getLogger(module_name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    is_child_logger = "." in module_name
    
    if not is_child_logger or force_file_handler:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = paths.logs / f"{module_name}_log_{timestamp}.txt"
        paths.logs.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    else:
        logger.propagate = True
    
    return logger

def get_calling_script_name():
    """
    Automatically detect the name of the script that's calling this function
    gets name of first frame in call stack (presumably the origin script)
    
    Returns:
        str: Name of the calling script without .py extension
    """
    current_file = os.path.abspath(__file__)
    
    # Use sys.argv[0] which is the script that was executed
    if hasattr(sys, "argv") and sys.argv and sys.argv[0]:
        main_script = os.path.abspath(sys.argv[0])
        if main_script != current_file and os.path.exists(main_script):
            script_name = os.path.basename(main_script)
            if script_name.endswith(".py"):
                script_name = script_name[:-3]
            return script_name
    
    return "unknown_caller"

def setup_child_logger(child_name: str, log_level=logging.DEBUG):
    """
    Create child logger that automatically fills in parent name
    e.g. setup_child_logger("data_processing") -> "initial_processing.data_processing"
    
    Args:
        child_name: Name for this specific module (e.g., "data_processing")
        log_level: Logging level for this logger
    
    Returns:
        logger: Child logger that propagates to the auto-detected parent
    """
    parent_name = get_calling_script_name()
    full_logger_name = f"{parent_name}.{child_name}"
    
    return setup_logger(full_logger_name, log_level)

def setup_script_logger(script_name: Optional[str] = None, log_level=logging.INFO):
    """
    Set up the main logger for a script
    Auto-detects script name if not provided
    
    Args:
        script_name: Name for the logger. If None, auto-detects.
        log_level: Logging level for the root logger
    
    Returns:
        logger: Root logger for the script
    """
    if script_name is None:
        script_name = get_calling_script_name()
    
    return setup_logger(script_name, log_level)

# =====================================
# Lazy initialization helper for modules
# =====================================

class LazyModuleLogger:
    """
    A logger wrapper that delays initialization until first use.
    This ensures the call stack detection happens at runtime, not import time.
    """
    def __init__(self, module_name: str, log_level=logging.DEBUG):
        self.module_name = module_name
        self.log_level = log_level
        self._logger = None
    
    def _get_logger(self):
        """Get the actual logger, creating it if necessary"""
        if self._logger is None:
            parent_name = get_calling_script_name()
            full_logger_name = f"{parent_name}.{self.module_name}"
            print(full_logger_name)
            self._logger = setup_logger(full_logger_name, self.log_level)
        return self._logger
    
    def info(self, message, *args, **kwargs):
        return self._get_logger().info(message, *args, **kwargs)
    
    def debug(self, message, *args, **kwargs):
        return self._get_logger().debug(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        return self._get_logger().warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        return self._get_logger().error(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        return self._get_logger().critical(message, *args, **kwargs)
    
    @property
    def name(self):
        return self._get_logger().name

def get_module_logger(module_name: str, log_level=logging.DEBUG):
    """
    Get a lazy-initialized module logger that auto-detects its parent at runtime
    
    Args:
        module_name: Name for this specific module (e.g., "data_processing")
        log_level: Logging level for this logger
    
    Returns:
        LazyModuleLogger: A logger that behaves like a normal logger but 
                         initializes the hierarchy when first used
    
    Example:
        # In data_processing.py:
        logger = get_module_logger("data_processing")
        
        # Logger hierarchy gets determined when you first call:
        logger.info("This message determines the parent at runtime!")
    """
    return LazyModuleLogger(module_name, log_level)