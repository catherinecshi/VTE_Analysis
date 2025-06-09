"""plots performance for different trial types for a single rat and day"""

from config import settings
from preprocessing import data_processing
from preprocessing import process_statescript
from visualization import performance_plots

RAT = "BP15"
DAY = "Day1"

data_structure = data_processing.load_data_structure()

ss_data = data_structure[RAT][DAY][settings.SS]
trial_types, total_trials, correct_trials = process_statescript.trial_accuracy(ss_data)

performance_plots.plot_trial_accuracy(total_trials, correct_trials, trial_types)
