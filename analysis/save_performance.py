import performance_analysis
from config import settings
from utilities import logging_utils

# pylint: disable=broad-exception-caught, logging-fstring-interpolation
logger = logging_utils.setup_script_logger()
logger.info("starting save_performance.py")

try:
    rat_performances = performance_analysis.get_all_rats_performance(save=True)
except Exception as e:
    logger.critical(f"error {e} when saving performance for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} and potentially {settings.CURRENT_TRIAL}")

try:
    performance_analysis.save_all_perf_changes(rat_performances)
except Exception as e:
    logger.critical(f"error {e} when saving perf change for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} and potentially {settings.CURRENT_TRIAL}")

try:
    performance_analysis.save_all_perf_changes_trials(rat_performances)
except Exception as e:
    logger.critical(f"error {e} when saving perf change for trials for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} and potentially {settings.CURRENT_TRIAL}")