"""plot all performances for all sessions and all days"""

from analysis import performance_analysis
from visualization import performance_plots
    
rats_performances = performance_analysis.get_all_rats_performance()
performance_plots.plot_rat_performance(rats_performances)