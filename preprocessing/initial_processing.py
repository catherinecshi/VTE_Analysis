from config.paths import paths
from preprocessing import data_processing
from utilities import logging_utils

logger = logging_utils.setup_script_logger()
logger.info("starting initial processing")

SAVE_PATH = paths.data / "VTE_Data"
BP07_PATH = "/Users/catpillow/Documents/Draft_Analysis/data/VTE_Data/BP07/inferenceTraining"
data_processing.initial_to_inference(BP07_PATH)
data_processing.convert_all_timestamps(SAVE_PATH)
data_processing.convert_all_statescripts(SAVE_PATH)
data_processing.concat_duplicates(SAVE_PATH)
