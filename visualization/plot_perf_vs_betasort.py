import numpy as np
import pandas as pd

from config.paths import paths
from utilities import logging_utils
from models import betasort_test
from models import betasort

# pylint: disable=consider-using-enumeration
logger = logging_utils.setup_script_logger()

base_path = paths.betasort_data

def compare_model_to_one_rat(all_data_df, rat, n_simulations=100, tau=0.01, xi=0.99, threshold=0.75, test=False):
    all_models = {}
    
    # track global stimuli states
    global_U = {}
    global_L = {}
    global_R = {}
    global_N = {}
    
    # store match rates for each day
    match_rates = []
    
    index = 0 # to keep track of vtes indices
    
    # process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        print(f"\n  Rat {rat} Day {day}", end="", flush=True)
        
        # Extract relevant data
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        #rewards = day_data["correct"].values
        
        # Identify which stimuli are present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1  # +1 because of 0-indexing
        
        # Initialize a new model for this day
        model = betasort_test.Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        # Transfer state from previous days
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
        
        model.uncertainty_history = [model.get_all_stimulus_uncertainties()]
        model.ROC_uncertainty_history = [model.get_all_ROC_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.U_history = [model.U.copy()]
        model.L_history = [model.L.copy()]
        
        # process the trials for today
        participant_choices = np.column_stack((chosen_idx, unchosen_idx))
        n_trials = len(participant_choices)
        matches = np.zeros(n_trials)
        vtes = day_data["VTE"]
        
        for t in range(n_trials):
            if t % 100 == 0 or (n_trials < 100 and t % 10 == 0):
                print(".", end="", flush=True)
            
            chosen_idx, other_idx = participant_choices[t]
            #reward = rewards[t]
            
            # Validate indices (just in case)
            if not (0 <= chosen_idx < n_stimuli) or not (0 <= other_idx < n_stimuli):
                print(f"Day {day}, Trial {t}: Invalid indices - chosen {chosen_idx} unchosen {other_idx}")
                continue

            # run multiple simulations to get choice probability
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose(chosen_idx, other_idx, vtes[index])
                model_choices[sim] = model_choice
            
            # see how well the model matches up with real choices
            model_match_rate = np.mean(model_choices == chosen_idx)
            matches[t] = model_match_rate

            # update model based on actual feedback
            reward = 1 if chosen_idx < other_idx else 0
            model.update(chosen_idx, other_idx, reward, model_match_rate, threshold=threshold)
            
            #print(t, chosen_idx, reward)
            index += 1
        
        # calculate cumulative match rate
        cumulative_match_rate = np.mean(matches)
        match_rates.append(cumulative_match_rate)
        
        # store all the stuff for this day
        all_models[day] = model

        # update global states
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
    
    # return the models
    final_day = max(all_models.keys())
    return all_models[final_day], all_models, match_rates