import os
import numpy as np
import pandas as pd

from models import helper
from models import betasort

data_path = os.path.join(helper.BASE_PATH, "processed_data", "data_for_model")

for rat in os.listdir(data_path):
    if "TH510" not in rat:
        continue

    rat_path = os.path.join(data_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if ".DS_Store" in file or "zIdPhi" in file:
                continue

            file_path = os.path.join(root, file)
            file_csv = pd.read_csv(file_path)
            day = file.split(".")[0]
            
            # reform into compatible form
            chosen_idx = file_csv["first"]
            unchosen_idx = file_csv["second"]
            rewards = file_csv["correct"]
            participant_choices = np.column_stack((chosen_idx, unchosen_idx))
            
            # check for n_stimuli
            max_stimuli = max(max(chosen_idx), max(unchosen_idx)) + 1 # bc 0-index
            unique_stimuli_chosen = len(set(chosen_idx))
            unique_stimuli_unchosen = len(set(unchosen_idx))
            n_stimuli_first = max(unique_stimuli_chosen, unique_stimuli_unchosen)
            n_stimuli = min(max_stimuli, n_stimuli_first)
            
            # put through betasort model
            #model = betasort.analyze_real_data(participant_choices, rewards, n_stimuli, rat, day, tau=0.05, xi=0.95)
            
            # plot the uncertainty and positioms
            all_indices = np.unique(np.concatenate([chosen_idx, unchosen_idx]))
            index_to_letter = {idx: chr(65 + i) for i, idx, in enumerate(all_indices)}
            stimulus_labels = [f"Stimulus {index_to_letter[i]}" for i in range(n_stimuli)]
            
            #betasort.plot_uncertainty(model, stimulus_labels)
            #betasort.plot_positions(model, stimulus_labels)
            
            # plot beta distributions
            #betasort.plot_beta_distributions(model, stimulus_labels=stimulus_labels)
            #betasort.plot_boundaries_history(model, stimulus_labels)
            
            # compare betasort with real deta
            #matches, _, _ = betasort.compare_model_to_rats(participant_choices, n_stimuli, rat, day, tau=0.05, xi=0.95)
            #betasort.plot_match_rates(matches)
            
            # find the best parameters for the data
            best_xi, best_tau, best_performance, param_performances = betasort.find_optimal_parameters(participant_choices, n_stimuli, rat, day)
            print(f"best xi, tau, performance: {best_xi}, {best_tau}, {best_performance}")
            
            betasort.parameter_performance_heatmap(param_performances)
            
            # plot best parameters
            best_model = betasort.analyze_real_data(participant_choices, rewards, n_stimuli, rat, day, tau=best_tau, xi=best_xi)
            betasort.plot_best_parameters(best_model)