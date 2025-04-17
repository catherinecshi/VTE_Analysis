import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# pylint: disable=global-statement, logging-fstring-interpolation, consider-using-enumerate

class Betasort:
    """
    Implementation of betasort model according to pseudocode from "Implicit Value Updating Explains TI Performance"
    Greg Jensen et al
    """
    
    def __init__(self, n_stimuli, rat, day, tau=0.05, xi=0.95):
        """
        initialise betasort model
        
        Parameters:
            - n_stimuli (int): number of stimuli in the list
            - tau (float): noise parameter
            - xi (float): recall parameter
        """
        
        self.n_stimuli = n_stimuli
        self.rat = rat
        self.day = day
        self.trial = 0
        self.tau = tau
        self.xi = xi
        
        # unexpected uncertainty
        self.seen_stimuli = set()
        self.unexpected_uncertainty = 0
        self.unexpected_uncertainty_decay = 0.9

        # initialise memory arrays
        self.U = np.ones(n_stimuli) # upper parameter
        self.L = np.ones(n_stimuli) # lower parameter
        self.R = np.ones(n_stimuli) # rewards
        self.N = np.ones(n_stimuli) # non rewards
        
        # store history for later plotting
        self.uncertainty_history = [self.get_all_stimulus_uncertainties()]
        self.relational_uncertainty_history = [self.get_all_relational_uncertainties()]
        self.ROC_uncertainty_history = [self.get_all_ROC_uncertainties()]
        self.unexpected_uncertainty_history = [0]
        self.position_history = [self.get_all_positions()]
        self.U_history = [self.U.copy()]
        self.L_history = [self.L.copy()]
    
    def choose(self, available_stimuli):
        """
        choice policy - selects stimuli from available choices
        won't be used when integrating real data
        """
        
        # generate random values for each available stimulus
        X = np.zeros(len(available_stimuli))
        
        for i, stim_idx in enumerate(available_stimuli):
            if np.random.random() < self.tau: # choose randomly
                X[i] = np.random.beta(1, 1)
            else: # base off of learned stimuli beta distributions
                X[i] = np.random.beta(self.U[stim_idx] + 1, self.L[stim_idx] + 1)
        
        # choose stimulus with largest value
        chosen_idx = np.argmax(X) 
        return available_stimuli[chosen_idx]

    def get_uncertainty_stimulus(self, stimulus_idx):
        """
        calculates uncertainty for a given stimulus using variance of beta distribution
        
        Parameters:
            - stimulus_idx (int): index of stimulus
        
        Returns:
            - (float): uncertainty value
        """
        
        a = self.U[stimulus_idx]
        b = self.L[stimulus_idx]
        
        # avoid division by zero or undefined values
        if a + b < 2:
            return 1.0 # maximum uncertainty
        
        return (a * b) / ((a + b)**2 * (a + b + 1))

    def get_all_stimulus_uncertainties(self):
        """Get uncertainty values for all stimuli"""
        return np.array([self.get_uncertainty_stimulus(i) for i in range(self.n_stimuli)])
    
    def get_uncertainty_relation(self, chosen_idx, other_idx, n_samples=1000):
        """
        Calculates the uncertainty about the relationship between two stimuli
        
        Parameters:
            - chosen_idx (int): index of first stimulus
            - other_idx (int): index of second stimulus
            - n_samples (int): number of Monte Carlo samples
            
        Returns:
            - (float): uncertainty value between 0 and 1 (0=certain, 1=maximally uncertain)
        """
        # Get Beta distribution parameters
        a1 = self.U[chosen_idx] + 1
        b1 = self.L[chosen_idx] + 1
        a2 = self.U[other_idx] + 1
        b2 = self.L[other_idx] + 1
        
        # Generate samples from both distributions
        samples1 = np.random.beta(a1, b1, n_samples)
        samples2 = np.random.beta(a2, b2, n_samples)
        
        # Calculate probability that samples1 > samples2
        prob_greater = np.mean(samples1 > samples2)
        
        # Convert to uncertainty measure (0 = certain, 1 = maximally uncertain)
        uncertainty = 1 - 2 * abs(prob_greater - 0.5)
        
        return uncertainty
    
    def get_all_relational_uncertainties(self):
        """get probabilistic uncertainties for all adjacent pairs of stimulus"""
        return np.array([self.get_uncertainty_relation(i, i+1) for i in range(self.n_stimuli-1)])

    def get_uncertainty_relation_ROC(self, chosen_idx, other_idx, n_samples=10000):
        """
        Calculates the uncertainty about the relationship between two stimuli using ROC analysis
        
        Parameters:
            - stimulus_idx1 (int): index of first stimulus
            - stimulus_idx2 (int): index of second stimulus
            - n_samples (int): number of samples for ROC calculation
            
        Returns:
            - (float): uncertainty value between 0 and 1 (0=certain, 1=maximally uncertain)
        """
        # Get Beta distribution parameters
        a1 = self.U[chosen_idx] + 1
        b1 = self.L[chosen_idx] + 1
        a2 = self.U[other_idx] + 1
        b2 = self.L[other_idx] + 1
        
        # Generate samples from both distributions
        samples1 = np.random.beta(a1, b1, n_samples)
        samples2 = np.random.beta(a2, b2, n_samples)
        
        # Calculate AUC (Area Under the ROC Curve)
        # This is equivalent to the probability that a randomly selected
        # value from samples1 exceeds a randomly selected value from samples2
        auc = 0
        for i in range(n_samples):
            auc = np.mean(samples1 > samples2)
        auc /= n_samples
        
        # Convert AUC to uncertainty measure
        # AUC of 0.5 represents maximum uncertainty (distributions completely overlap)
        # AUC of 0 or 1 represents minimum uncertainty (distributions completely separated)
        uncertainty = 1 - 2 * abs(auc - 0.5)
        
        return uncertainty
    
    def get_all_ROC_uncertainties(self):
        """get probabilistic uncertainties for all adjacent pairs of stimulus"""
        return np.array([self.get_uncertainty_relation_ROC(i, i+1) for i in range(self.n_stimuli-1)])
    
    def get_all_positions(self):
        """Get estimated positions for all stimuli"""
        positions = np.zeros(self.n_stimuli)
        for i in range(self.n_stimuli):
            if self.U[i] + self.L[i] == 0:
                positions[i] = 0.5
            else:
                positions[i] = self.U[i] / (self.U[i] + self.L[i])
        return positions
        
    def thresholding_update(self, chosen, unchosen, reward, probability, threshold):
        """
        implements first updating policy
        
        Parameters:
            - chosen : int
                - index of chosen stimulus
            - unchosen : int
                - index of unchosen stimulus
            - reward : int
                - 1 for reward and 0 otherwise
            - probability : float
                - probability of how much the simulated data matches up with the real choice
                - used to update the model proportionally
            - threshold : float
                - threshold for above which consolidation is done
        """
        
        # update trial count
        self.trial += 1
        
        # relax
        self.R = self.R * self.xi
        self.N = self.N * self.xi
        
        # estimate trial reward rate
        E = self.R / (self.R + self.N)
        xi_R = E / (E + 1) + 0.5
        
        # relax some more :)
        self.U = self.U * xi_R * self.xi
        self.L = self.L * xi_R * self.xi

        # dynamic version of updating values by checking how much model matches with real data
        V = self.U / (self.U + self.L)

        if reward == 1 and probability < threshold:
            # update reward rate
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            # radical new moves
            self.U[chosen] = self.U[chosen] + (1)
            self.L[unchosen] = self.L[unchosen] + (1)
        elif reward == 1: 
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            self.U = self.U + V
            self.L = self.L + (1 - V)
        else:
            # update reward rate
            self.N[unchosen] = self.N[unchosen] + 1
            self.N[chosen] = self.N[chosen] + 1
            
            # shift unchosen up, chosen down
            self.U[unchosen] = self.U[unchosen] + 1
            self.L[chosen] = self.L[chosen] + 1
            
            # process other stimuli (implicit inference)
            for j in range(self.n_stimuli):
                if j != chosen and j != unchosen:
                    if V[j] > V[chosen] and V[j] < V[unchosen]:
                        # j fell between chosen and unchosen (consolidate)
                        self.U[j] = self.U[j] + V[j]
                        self.L[j] = self.L[j] + (1 - V[j])
                    elif V[j] < V[unchosen]:
                        # shift j down
                        self.L[j] = self.L[j] + 1
                    elif V[j] > V[chosen]:
                        # shift j up
                        self.U[j] = self.U[j] + 1

        # store updated uncertainties and positions upper and lower
        self.uncertainty_history.append(self.get_all_stimulus_uncertainties())
        self.relational_uncertainty_history.append(self.get_all_relational_uncertainties())
        self.ROC_uncertainty_history.append(self.get_all_ROC_uncertainties())
        self.position_history.append(self.get_all_positions())
        self.U_history.append(self.U.copy())
        self.L_history.append(self.L.copy())
    
    def UU_update(self, chosen, unchosen, reward, probability, threshold):
        """
        implements first updating policy
        
        Parameters:
            - chosen : int
                - index of chosen stimulus
            - unchosen : int
                - index of unchosen stimulus
            - reward : int
                - 1 for reward and 0 otherwise
            - probability : float
                - probability of how much the simulated data matches up with the real choice
                - used to update the model proportionally
            - threshold : float
                - threshold for above which consolidation is done
        """
        
        # update trial count
        self.trial += 1
        
        # unexpected uncertainty based on whether new stimuli encountered
        new_stimuli_encountered = False
        
        if chosen not in self.seen_stimuli:
            self.seen_stimuli.add(chosen)
            new_stimuli_encountered = True
        
        if unchosen not in self.seen_stimuli:
            self.seen_stimuli.add(unchosen)
            new_stimuli_encountered = True
        
        # update unexpected uncertainty
        if new_stimuli_encountered:
            self.unexpected_uncertainty = 1.0
        else:
            self.unexpected_uncertainty *= self.unexpected_uncertainty_decay
        
        # unexpected uncertainty causes more forgetting
        effective_xi = 1 - 0.5 * self.unexpected_uncertainty
        
        # relax
        self.R = self.R * effective_xi
        self.N = self.N * effective_xi
        
        # estimate trial reward rate
        E = self.R / (self.R + self.N)
        xi_R = E / (E + 1) + 0.5
        
        # relax some more :)
        self.U = self.U * xi_R * effective_xi
        self.L = self.L * xi_R * effective_xi

        # dynamic version of updating values by checking how much model matches with real data
        V = self.U / (self.U + self.L)

        if reward == 1 and probability < threshold:
            # update reward rate
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            # radical new moves
            self.U[chosen] = self.U[chosen] + (1 - probability)
            self.L[unchosen] = self.L[unchosen] + (1 - probability)
        elif reward == 1: 
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            self.U = self.U + V
            self.L = self.L + (1 - V)
        else:
            # update reward rate
            self.N[unchosen] = self.N[unchosen] + 1
            self.N[chosen] = self.N[chosen] + 1
            
            # shift unchosen up, chosen down
            self.U[unchosen] = self.U[unchosen] + 1
            self.L[chosen] = self.L[chosen] + 1
            
            # process other stimuli (implicit inference)
            for j in range(self.n_stimuli):
                if j != chosen and j != unchosen:
                    if V[j] > V[chosen] and V[j] < V[unchosen]:
                        # j fell between chosen and unchosen (consolidate)
                        self.U[j] = self.U[j] + V[j]
                        self.L[j] = self.L[j] + (1 - V[j])
                    elif V[j] < V[unchosen]:
                        # shift j down
                        self.L[j] = self.L[j] + 1
                    elif V[j] > V[chosen]:
                        # shift j up
                        self.U[j] = self.U[j] + 1

        # store updated uncertainties and positions upper and lower
        self.uncertainty_history.append(self.get_all_stimulus_uncertainties())
        self.relational_uncertainty_history.append(self.get_all_relational_uncertainties())
        self.ROC_uncertainty_history.append(self.get_all_ROC_uncertainties())
        self.unexpected_uncertainty_history.append(self.unexpected_uncertainty)
        self.position_history.append(self.get_all_positions())
        self.U_history.append(self.U.copy())
        self.L_history.append(self.L.copy())

class Betasort_Lite:
    """
    Lightweight implementation of Betasort model for optimization.
    Removes unnecessary uncertainty calculations and history tracking.
    """
    
    def __init__(self, n_stimuli, tau=0.05, xi=0.95):
        self.n_stimuli = n_stimuli
        self.tau = tau
        self.xi = xi
        
        # Initialize memory arrays (without tracking histories)
        self.U = np.ones(n_stimuli)  # upper parameter
        self.L = np.ones(n_stimuli)  # lower parameter
        self.R = np.ones(n_stimuli)  # rewards
        self.N = np.ones(n_stimuli)  # non rewards
    
    def choose(self, available_stimuli):
        """Choice policy - selects stimuli from available choices"""
        # Generate random values for each available stimulus
        X = np.zeros(len(available_stimuli))
        
        for i, stim_idx in enumerate(available_stimuli):
            if np.random.random() < self.tau:  # choose randomly
                X[i] = np.random.beta(1, 1)
            else:  # base off of learned stimuli beta distributions
                X[i] = np.random.beta(self.U[stim_idx] + 1, self.L[stim_idx] + 1)
        
        # Choose stimulus with largest value
        chosen_idx = np.argmax(X) 
        return available_stimuli[chosen_idx]
    
    def update(self, chosen, unchosen, reward, probability, threshold=0.75):
        """Simplified update method without tracking histories"""
        # Relax
        self.R = self.R * self.xi
        self.N = self.N * self.xi
        
        # Estimate trial reward rate
        E = self.R / (self.R + self.N)
        xi_R = E / (E + 1) + 0.5
        
        # Relax some more
        self.U = self.U * xi_R * self.xi
        self.L = self.L * xi_R * self.xi

        # Get value estimates
        V = self.U / (self.U + self.L)

        if reward == 1 and probability < threshold:
            # Update reward rate
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            # Update model
            self.U[chosen] = self.U[chosen] + 1
            self.L[unchosen] = self.L[unchosen] + 1
        elif reward == 1: 
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            self.U = self.U + V
            self.L = self.L + (1 - V)
        else:
            # Update reward rate
            self.N[unchosen] = self.N[unchosen] + 1
            self.N[chosen] = self.N[chosen] + 1
            
            # Shift unchosen up, chosen down
            self.U[unchosen] = self.U[unchosen] + 1
            self.L[chosen] = self.L[chosen] + 1
            
            # Process other stimuli (implicit inference)
            for j in range(self.n_stimuli):
                if j != chosen and j != unchosen:
                    if V[j] > V[chosen] and V[j] < V[unchosen]:
                        # j fell between chosen and unchosen (consolidate)
                        self.U[j] = self.U[j] + V[j]
                        self.L[j] = self.L[j] + (1 - V[j])
                    elif V[j] < V[unchosen]:
                        # Shift j down
                        self.L[j] = self.L[j] + 1
                    elif V[j] > V[chosen]:
                        # Shift j up
                        self.U[j] = self.U[j] + 1

def compare_model_to_one_rat_lite(all_data_df, rat, n_simulations=50, tau=0.01, xi=0.99, threshold=0.75, verbose=False):
    """
    Lightweight version of compare_model_to_one_rat that omits unnecessary calculations
    and only returns match rates needed for optimization.
    """
    # Track global stimuli states
    global_U = {}
    global_L = {}
    global_R = {}
    global_N = {}
    
    # Store match rates for each day
    match_rates = []
    
    # Process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        if verbose and day % 5 == 0:  # Print only every 5th day to reduce output
            print(f"Day {day}", end=" ", flush=True)
        
        # Extract relevant data
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        
        # Identify which stimuli are present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1  # +1 because of 0-indexing
        
        # Initialize a new model for this day
        model = Betasort_Lite(n_stimuli, tau=tau, xi=xi)
        
        # Transfer state from previous days
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
        
        # Process the trials for today
        participant_choices = np.column_stack((chosen_idx, unchosen_idx))
        n_trials = len(participant_choices)
        matches = np.zeros(n_trials)
        
        for t in range(n_trials):
            chosen_idx, other_idx = participant_choices[t]
            
            # Skip invalid indices
            if not (0 <= chosen_idx < n_stimuli) or not (0 <= other_idx < n_stimuli):
                continue

            # Run multiple simulations to get choice probability
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx, other_idx])
                model_choices[sim] = model_choice
            
            # Calculate match rate
            model_match_rate = np.mean(model_choices == chosen_idx)
            matches[t] = model_match_rate

            # Update model based on actual feedback
            reward = 1 if chosen_idx < other_idx else 0
            model.update(chosen_idx, other_idx, reward, model_match_rate, threshold=threshold)
        
        # Calculate cumulative match rate
        cumulative_match_rate = np.mean(matches)
        match_rates.append(cumulative_match_rate)
        
        if verbose and day % 5 == 0:
            print(f"Match rate: {cumulative_match_rate:.4f}")
        
        # Update global states
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
    
    # Only return match rates - this is all we need for optimization
    return match_rates

def compare_model_to_one_rat(all_data_df, rat, n_simulations=100, tau=0.01, xi=0.99, threshold=0.75):
    all_models = {}
    
    # track global stimuli states
    global_U = {}
    global_L = {}
    global_R = {}
    global_N = {}
    
    # store match rates for each day
    match_rates = []
    
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
        model = Betasort(n_stimuli, rat, day)
        
        # Transfer state from previous days
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
        
        model.uncertainty_history = [model.get_all_stimulus_uncertainties()]
        model.relational_uncertainty_history = [model.get_all_relational_uncertainties()]
        model.ROC_uncertainty_history = [model.get_all_ROC_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.U_history = [model.U.copy()]
        model.L_history = [model.L.copy()]
        
        # process the trials for today
        participant_choices = np.column_stack((chosen_idx, unchosen_idx))
        n_trials = len(participant_choices)
        matches = np.zeros(n_trials)
        
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
                model_choice = model.choose([chosen_idx, other_idx])
                model_choices[sim] = model_choice
            
            # see how well the model matches up with real choices
            model_match_rate = np.mean(model_choices == chosen_idx)
            matches[t] = model_match_rate

            # update model based on actual feedback
            reward = 1 if chosen_idx < other_idx else 0
            model.thresholding_update(chosen_idx, other_idx, reward, model_match_rate, threshold=threshold)
            
            #print(t, chosen_idx, reward)
        
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

def binomial_analysis_by_session(all_data_df, rat, tau=0.05, xi=0.95, n_simulations=100):
    """Run binomial test to check if model predictions are significantly similar to rat choices"""
    
    all_data_df = all_data_df.sort_values('Day')
    
    session_results = {}
    for day, day_data in all_data_df.groupby('Day'):
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        # initialize model with current parameters
        model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        # These track correctness
        model_correct = np.zeros(len(chosen_idx))
        rat_correct = np.zeros(len(chosen_idx))
        
        # NEW: This will track how often model agrees with rat
        model_rat_matches = np.zeros(len(chosen_idx))
        
        for t in range(len(chosen_idx)):
            reward = 1 if chosen_idx[t] < unchosen_idx[t] else 0
            sim_correct = np.zeros(n_simulations)
            model_choices = np.zeros(n_simulations)
            
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx[t], unchosen_idx[t]])
                model_choices[sim] = model_choice
                sim_correct[sim] = 1 if model_choice == min(chosen_idx[t], unchosen_idx[t]) else 0
            
            # see how well the model matches up with real choices
            current_model_match_rate = np.mean(model_choices == chosen_idx[t])
            
            # NEW: Store whether model and rat agreed on this trial
            model_rat_matches[t] = current_model_match_rate
            
            model_correct[t] = np.mean(sim_correct)
            rat_correct[t] = reward
            model.UU_update(chosen_idx[t], unchosen_idx[t], reward, current_model_match_rate, threshold=0.6)
        
        # NEW: binomial test on model-rat agreement instead of correctness
        # Count total matches across all trials
        n_matches = int(np.sum(model_rat_matches * n_simulations))
        # Total number of simulation runs
        n_trials = len(model_rat_matches) * n_simulations
        # Null hypothesis: random chance of matching (0.5 for binary choice)
        chance_match_rate = 0.5  # If there are only 2 choices in each trial
        
        # Test if match rate is significantly higher than chance
        p_value = stats.binomtest(n_matches, n_trials, p=chance_match_rate, alternative='greater')
        
        session_results[day] = {
            'matches': n_matches,
            'trials': n_trials,
            'match_rate': n_matches/n_trials,
            'p_value': p_value,
            'chance_rate': chance_match_rate,
            # NEW: Now significant when p < 0.05, indicating match rate better than chance
            'significant': p_value.pvalue < 0.05  # Significant if model-rat agreement is BETTER than chance
        }
    
    return session_results

def t_test_model_vs_real_choices(all_data_df, rat, tau=0.05, xi=0.95, n_simulations=100):
    """
    Perform t-tests to compare model choices with real choices across sessions.
    
    Parameters:
        - all_data_df : DataFrame
            - data for all days for a single rat
        - rat : String
            - the rat ID
        - tau, xi : model parameters
        - n_simulations : number of simulations per trial
        
    Returns:
        - session_results : dict
            - contains test results for each day
        - overall_results : dict
            - contains test results across all days
    """
    all_data_df = all_data_df.sort_values('Day')
    
    session_results = {}
    all_model_choices = []
    all_real_choices = []
    
    for day, day_data in all_data_df.groupby('Day'):
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        # Initialize model for this day
        model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        # Track model choices and rat choices for this day
        day_model_choices = []
        day_real_choices = []
        
        for t in range(len(chosen_idx)):
            # Get the rat's choice
            real_choice = chosen_idx[t]
            
            # Simulate model choice multiple times and take the most frequent choice
            model_choices_count = {chosen_idx[t]: 0, unchosen_idx[t]: 0}
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx[t], unchosen_idx[t]])
                model_choices[sim] = model_choice
                model_choices_count[model_choice] += 1
            
            current_model_match_rate = np.mean(model_choices == chosen_idx[t])
            
            # Determine the model's preferred choice
            model_choice = chosen_idx[t] if model_choices_count[chosen_idx[t]] > model_choices_count[unchosen_idx[t]] else unchosen_idx[t]
            
            day_model_choices.append(model_choice)
            day_real_choices.append(real_choice)
            all_model_choices.append(model_choice)
            all_real_choices.append(real_choice)
            
            # Update model based on actual feedback
            reward = 1 if chosen_idx[t] < unchosen_idx[t] else 0
            model.thresholding_update(chosen_idx[t], unchosen_idx[t], reward, current_model_match_rate, threshold=0.6)
        
        # Convert to arrays
        day_model_choices = np.array(day_model_choices)
        day_real_choices = np.array(day_real_choices)
        
        # make matches the match_model_rate
        # paired 2 way t test in fuure (MAKE SURE PAIRED)
        # Perform paired t-test for this day
        # (comparing whether the model and rat chose the same stimulus)
        matches = (day_model_choices == day_real_choices).astype(int)
        t_stat, p_value = stats.ttest_1samp(matches, 0.5)  # Test if match rate is significantly different from chance
        
        session_results[day] = {
            'n_trials': len(matches),
            'match_rate': np.mean(matches),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Overall test across all days
    all_matches = (np.array(all_model_choices) == np.array(all_real_choices)).astype(int)
    overall_t_stat, overall_p_value = stats.ttest_1samp(all_matches, 0.5)
    
    overall_results = {
        'n_trials': len(all_matches),
        'match_rate': np.mean(all_matches),
        't_statistic': overall_t_stat,
        'p_value': overall_p_value,
        'significant': overall_p_value < 0.05
    }
    
    return session_results, overall_results

def find_optimal_parameters_for_rat(all_data_df, rat, n_simulations=100, xi_values=None, tau_values=None):
    if xi_values is None:
        xi_values_coarse = np.arange(0.75, 0.99 + 0.01, 0.01)
        xi_values_fine = np.arange(0.99, 1.0 + 0.001, 0.001)
        
        # Concatenate arrays and remove any duplicates (0.99 appears in both)
        xi_values = np.unique(np.concatenate([xi_values_coarse, xi_values_fine]))

    if tau_values is None:
        tau_values_fine = np.arange(0.001, 0.01 + 0.001, 0.001)
        tau_values_coarse = np.arange(0.01, 0.25 + 0.01, 0.01)
        
        # Concatenate arrays and remove any duplicates (0.01 appears in both)
        tau_values = np.unique(np.concatenate([tau_values_fine, tau_values_coarse]))
    
    param_performances = {}
    
    # for each parameter combination
    for xi in xi_values:
        for tau in tau_values:
            # start a fresh model with these parameters
            _, _, matches_rate = compare_model_to_one_rat(all_data_df, rat, tau=tau, xi=xi)
            
            param_performances[(xi, tau)] = matches_rate # store rate
    
    # find best parameter combination
    avg_performances = {params: np.mean(day_rates) for params, day_rates in param_performances.items()}
    
    best_params = max(avg_performances, key=avg_performances.get)
    best_xi, best_tau = best_params
    best_performance = avg_performances[best_params]
    
    return best_xi, best_tau, best_performance, param_performances

def find_optimal_threshold(all_data_df, rat, n_simulations=100, xi_values=None, tau_values=None, thresholds=None):
    if xi_values is None:
        xi_values = np.arange(0.9, 1, 0.001)

    if tau_values is None:
        tau_values = np.arange(0, 0.1, 0.001)
    
    if thresholds is None:
        thresholds = np.arange(0.4, 0.7, 0.01)
    
    param_performances = {}
    all_xi = []
    all_tau = []
    all_threshold = []
    all_performances = []
    all_days = []
    
    # list of unique days for labeling later
    unique_days = sorted(all_data_df['Day'].unique())
    num_days = len(unique_days)
    
    # for each parameter combination
    for xi in xi_values:
        for tau in tau_values:
            for threshold in thresholds:
                # start a fresh model with these parameters
                try:
                    _, _, matches_rate = compare_model_to_one_rat(all_data_df, 
                                                                  rat, 
                                                                  tau=tau, 
                                                                  xi=xi, 
                                                                  threshold=threshold)
                    
                    param_performances[(xi, tau, threshold)] = matches_rate # store rate
                    
                    # add list for dataframe
                    for day_idx, day in enumerate(unique_days):
                        all_xi.append(xi)
                        all_tau.append(tau)
                        all_threshold.append(threshold)
                        all_days.append(day)
                        all_performances.append(matches_rate[day_idx])
                
                except Exception as e:
                    print(f"Error with parameters xi={xi}, tau={tau}, threshold={threshold}: {e}")
                    continue
    
    # create dataframe
    results_df = pd.DataFrame({
        'xi': all_xi,
        'tau': all_tau,
        'threshold': all_threshold,
        'day': all_days,
        'match_rate': all_performances
    })
    
    # find best parameter combination
    avg_performances = {}
    for params, day_rates in param_performances.items():
        avg_performances[params] = np.mean(day_rates)
    
    # add a column for average performances across days
    avg_df = results_df.groupby(['xi', 'tau', 'threshold'])['match_rate'].mean().reset_index()
    avg_df['day'] = 'average'
    avg_df = avg_df.rename(columns={'match_rate': 'avg_match_rate'})
    
    # separate dataframe for just the summary
    summary_df = pd.DataFrame({
        'best_xi': [best_xi],
        'best_tau': [best_tau],
        'best_threshold': [best_threshold],
        'best_performance': [best_performance]
    })
    
    best_params = max(avg_performances, key=avg_performances.get)
    best_xi, best_tau, best_threshold = best_params
    best_performance = avg_performances[best_params]
    
    return best_xi, best_tau, best_threshold, best_performance, param_performances, results_df, summary_df

def model_performance_objective(params, *args):
    """Optimization objective function using lightweight model"""
    # Unpack parameters
    xi, tau, threshold = params
    all_data, rat, verbose = args
    
    if verbose:
        #print(f"Evaluating: xi={xi:.6f}, tau={tau:.6f}, threshold={threshold:.6f}", end="", flush=True)
        start_time = time.time()
    
    # Evaluate model with these parameters
    match_rates = compare_model_to_one_rat_lite(
        all_data, rat,
        tau=tau, xi=xi, threshold=threshold,
        verbose=False  # Set to True to see per-day progress
    )
    
    # Calculate performance
    performance = np.mean(match_rates)
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f" → Performance: {performance:.6f}, Time: {elapsed_time:.2f}s")
    
    # Return negative match rate since we're minimizing
    return -performance

def diff_evolution_lite(all_data, rat, verbose=True, max_iter=100, popsize=25):
    """Optimized differential evolution using lightweight model"""
    if verbose:
        print("\n" + "="*50)
        print(f"Starting differential evolution at {time.strftime('%H:%M:%S')}")
        print(f"Parameter bounds: xi=[0.75, 0.999], tau=[0.001, 0.25], threshold=[0.6, 0.9]")
        print(f"Population size: {popsize}, Max iterations: {max_iter}")
        print("="*50 + "\n")
    
    total_start_time = time.time()
    
    # Store iteration data for tracking
    iterations = [0]
    last_time = [time.time()]
    
    def callback(xk, convergence):
        iterations[0] += 1
        current_time = time.time()
        iter_time = current_time - last_time[0]
        last_time[0] = current_time
        
        best_xi, best_tau, best_threshold = xk
        performance = -model_performance_objective(xk, all_data, rat, False)
        
        if verbose:
            print(f"\nIteration {iterations[0]} completed in {iter_time:.2f}s")
            print(f"  Best parameters: xi={best_xi:.6f}, tau={best_tau:.6f}")
            print(f"  Best performance: {performance:.6f}")
            print(f"  Convergence measure: {convergence:.6f}")
            
            # Estimate remaining time
            elapsed = time.time() - total_start_time
            time_per_iter = elapsed / iterations[0]
            remaining_iters = max_iter - iterations[0]
            est_remaining = time_per_iter * remaining_iters
            print(f"  Estimated remaining time: {est_remaining:.2f}s (~{est_remaining/60:.1f}m)")
        
        return False  # Don't stop the optimization
    
    result = differential_evolution(
        model_performance_objective,
        bounds=[(0.9, 0.999), (0.001, 0.1), (0.6, 0.9)],
        args=(all_data, rat, verbose),
        popsize=popsize,
        maxiter=max_iter,
        mutation=(0.5, 1.5),
        recombination=0.7,
        disp=verbose,
        callback=callback if verbose else None
    )
    
    total_time = time.time() - total_start_time
    
    best_xi, best_tau, best_threshold = result.x
    best_performance = -result.fun
    
    if verbose:
        print("\n" + "="*50)
        print(f"Differential evolution completed at {time.strftime('%H:%M:%S')}")
        print(f"Total optimization time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Total function evaluations: {result.nfev}")
        print(f"Best parameters: xi={best_xi:.6f}, tau={best_tau:.6f}, threshold={best_threshold:.6f}")
        print(f"Best performance: {best_performance:.6f}")
        print("="*50)
    
    return best_xi, best_tau, best_threshold, best_performance

def check_transitive_inference(model, n_simulations=100):
    """
    check over all decision probabilities for each possible choice
    
    Parameters:
    - model : Betasort
        - finished model
    - n_simulations : Int
        - number of simulations for model
    
    Returns:
    - results : {(int, int) : float}
        - chosen_idx, other_idx : % of getting it correct
    """
    
    # check its probabiltiies on what it would choose in transitive inference choices
    results = {}
    
    for chosen_idx in range(0, 4): # max out at 3
        for other_idx in range(chosen_idx + 1, 5):
            # skip same element
            if chosen_idx == other_idx:
                continue
            
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx, other_idx])
                model_choices[sim] = model_choice
            
            # see how well the model matches up with real choices
            model_match_rate = np.mean(model_choices == chosen_idx)
            
            results[(chosen_idx, other_idx)] = model_match_rate
            
    return results

def analyze_vte_uncertainty(all_data_df, rat, tau=0.05, xi=0.95, threshold=0.6, n_simulations=100):
    """Analyze how VTEs correlate with different types of uncertainty

    Args:
        all_data_df (DataFrame): rodent choice data
        vte_data (DataFrame): VTE data:
        
        rat (string): identifier
        tau, xi, threshold (float): model parameters

    Returns:
        pair_vte_data: DataFrame wtih paired VTE & uncertainty
        all_models: Dictionary of models for each day
    """
    pair_vte_data = []
    
    all_models = {}
    global_U, global_L, global_R, global_N = {}, {}, {}, {}
    
    for day, day_data in all_data_df.groupby('Day'):
        # extract relevant data from choice dataset
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        rewards = day_data["correct"].values
        traj_nums = day_data["ID"].values
        vtes = day_data["VTE"].values
        
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        # transfer states from previous day
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
            else:
                print(f"stim idx {stim_idx} not found in global updates dict?? {global_U} on day {day}")
            
        # Initialize histories
        model.uncertainty_history = [model.get_all_stimulus_uncertainties()]
        model.relational_uncertainty_history = [model.get_all_relational_uncertainties()]
        model.ROC_uncertainty_history = [model.get_all_ROC_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.U_history = [model.U.copy()]
        model.L_history = [model.L.copy()]
        
        for t in range(len(chosen_idx)):
            traj_num = traj_nums[t]
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            vte = vtes[t]
            
            # get uncertainty before updates - uncertainty at time of choice
            # individual stimulus uncertainties
            stim1_uncertainty = model.get_uncertainty_stimulus(min(chosen, unchosen))
            stim2_uncertainty = model.get_uncertainty_stimulus(max(chosen, unchosen))
            
            # Get relational uncertainty between the specific pair
            pair_relational_uncertainty = model.get_uncertainty_relation(min(chosen, unchosen), max(chosen, unchosen))
            pair_roc_uncertainty = model.get_uncertainty_relation_ROC(min(chosen, unchosen), max(chosen, unchosen))
            
            vte_occurred = 0
            if vte:
                vte_occurred = 1
            
            # Store the pair-specific data
            pair_vte_data.append({
                'day': day,
                'trial_num': traj_num,
                'stim1': min(chosen, unchosen),
                'stim2': max(chosen, unchosen),
                'chosen': chosen,
                'unchosen': unchosen,
                'vte_occurred': vte_occurred,
                'stim1_uncertainty': stim1_uncertainty,
                'stim2_uncertainty': stim2_uncertainty,
                'pair_relational_uncertainty': pair_relational_uncertainty,
                'pair_roc_uncertainty': pair_roc_uncertainty,
                'reward': reward
            })
            
            # run multiple simulations to get choice probability
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen, unchosen])
                model_choices[sim] = model_choice
            
            # see how well the model matches up with real choices
            model_match_rate = np.mean(model_choices == chosen)
            
            # Update the model after recording uncertainty values
            model.thresholding_update(chosen, unchosen, reward, model_match_rate, threshold=threshold)
        
        # Store model for this day
        all_models[day] = model
        
        # Update global states for the next day
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
    
    # Convert to DataFrame for easier analysis
    pair_vte_df = pd.DataFrame(pair_vte_data)
    
    # Return the data and models
    return pair_vte_df, all_models

def analyze_correlations(pair_vte_df):
    """
    Analyzes correlations between VTE and uncertainty measures for each stimulus pair
    
    Parameters:
        - pair_vte_df: DataFrame with paired VTE and uncertainty values
        
    Returns:
        - results: Dictionary of correlation results by pair
    """
    from scipy.stats import pointbiserialr
    import statsmodels.api as sm
    
    # Identify all unique stimulus pairs
    pair_vte_df['pair'] = pair_vte_df.apply(lambda row: f"{row['stim1']}-{row['stim2']}", axis=1)
    unique_pairs = pair_vte_df['pair'].unique()
    
    # Initialize results dictionary
    results = {
        'overall': {},
        'by_pair': {},
        'by_uncertainty_measure': {}
    }
    
    # Overall correlations (across all pairs)
    uncertainty_measures = [
        'stim1_uncertainty', 
        'stim2_uncertainty', 
        'pair_relational_uncertainty', 
        'pair_roc_uncertainty'
    ]
    
    # Calculate overall correlations
    for measure in uncertainty_measures:
        r, p = pointbiserialr(pair_vte_df['vte_occurred'], pair_vte_df[measure])
        results['overall'][measure] = {'r': r, 'p': p}
    
    # Calculate correlations for each stimulus pair
    for pair in unique_pairs:
        pair_data = pair_vte_df[pair_vte_df['pair'] == pair]
        
        # Skip pairs with very few data points or no VTE variation
        if len(pair_data) < 5 or pair_data['vte_occurred'].nunique() < 2:
            results['by_pair'][pair] = "Insufficient data"
            continue
        
        pair_results = {}
        for measure in uncertainty_measures:
            try:
                r, p = pointbiserialr(pair_data['vte_occurred'], pair_data[measure])
                pair_results[measure] = {'r': r, 'p': p}
            except:
                pair_results[measure] = "Computation failed"
        
        # Logistic regression for this pair
        try:
            X = sm.add_constant(pair_data[uncertainty_measures])
            logit_model = sm.Logit(pair_data['vte_occurred'], X)
            logit_result = logit_model.fit(disp=0)
            
            pair_results['logistic_regression'] = {
                'params': logit_result.params.to_dict(),
                'pvalues': logit_result.pvalues.to_dict(),
                'pseudo_r2': logit_result.prsquared
            }
        except:
            pair_results['logistic_regression'] = "Computation failed"
        
        results['by_pair'][pair] = pair_results
    
    # Analyze by uncertainty measure (which measure best predicts VTE across all pairs)
    for measure in uncertainty_measures:
        # Compare VTE vs non-VTE trials for this measure
        vte_trials = pair_vte_df[pair_vte_df['vte_occurred'] == 1]
        non_vte_trials = pair_vte_df[pair_vte_df['vte_occurred'] == 0]
        
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(
            vte_trials[measure], 
            non_vte_trials[measure], 
            equal_var=False
        )
        
        results['by_uncertainty_measure'][measure] = {
            'mean_vte_trials': vte_trials[measure].mean(),
            'mean_non_vte_trials': non_vte_trials[measure].mean(),
            'difference': vte_trials[measure].mean() - non_vte_trials[measure].mean(),
            't_stat': t_stat,
            'p_value': p_value
        }
    
    return results




### PLOTTING FUNCTIONS -------------------------------------------------------------------------------------
def plot_stimulus_uncertainty(model, stimulus_labels=None, save=None):
    """
    plot uncertainty over trials
    
    Parameters:
        - model : Betasort
            - fitted model with uncertainty history
        - stimulus_labels : list, optional
            - labels for stimuli
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
        
    # convert history to array
    uncertainty_array = np.array(model.uncertainty_history)
    
    # plot
    plt.figure(figsize=(12, 8))
    
    for i in range(model.n_stimuli):
        plt.plot(uncertainty_array[:, i], label=stimulus_labels[i])
    
    plt.xlabel("Trial")
    plt.ylabel("Uncertainty (Variance)")
    plt.title("Trial-by-Trial Uncertainty from Betasort Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    
def plot_relational_uncertainty(model, stimulus_labels=None, save=None):
    """
    Plot probabilistic-based relational uncertainties between stimuli pairs
    
    Parameters:
        - model : Betasort
            - fitted model with uncertainty history
        - stimulus_labels : list, optional
            - labels for stimuli
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}-{i+1}" for i in range(model.n_stimuli-1)]
        
    # convert history to array
    uncertainty_array = np.array(model.relational_uncertainty_history)
    
    # plot
    plt.figure(figsize=(12, 8))
    
    for i in range(model.n_stimuli-1):
        plt.plot(uncertainty_array[:, i], label=stimulus_labels[i])
    
    plt.xlabel("Trial")
    plt.ylabel("Uncertainty (Variance)")
    plt.title("Trial-by-Trial Probabilistic Uncertainty from Betasort Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_ROC_uncertainty(model, stimulus_labels=None, save=None):
    """
    Plot probabilistic-based relational uncertainties between stimuli pairs
    
    Parameters:
        - model : Betasort
            - fitted model with uncertainty history
        - stimulus_labels : list, optional
            - labels for stimuli
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}-{i+1}" for i in range(model.n_stimuli-1)]
        
    # convert history to array
    uncertainty_array = np.array(model.ROC_uncertainty_history)
    
    # plot
    plt.figure(figsize=(12, 8))
    
    for i in range(model.n_stimuli-1):
        plt.plot(uncertainty_array[:, i], label=stimulus_labels[i])
    
    plt.xlabel("Trial")
    plt.ylabel("Uncertainty (Variance)")
    plt.title("Trial-by-Trial ROC Uncertainty from Betasort Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_relational_uncertainty_history(model, uncertainty_type='both', stimulus_labels=None, figsize=(15, 10), save=None):
    """
    Plot the history of relational uncertainties between stimuli pairs
    
    Parameters:
        - model : Betasort
            - fitted model with prob_uncertainty_relation_history and roc_uncertainty_relation_history
        - uncertainty_type : str
            - 'both', 'prob', or 'roc' to specify which uncertainty measure to plot
        - stimulus_labels : list, optional
            - labels for stimuli
        - figsize : tuple
            - figure size for the plot
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    # Determine which uncertainty types to plot
    plot_prob = uncertainty_type in ['both', 'prob']
    plot_roc = uncertainty_type in ['both', 'roc']
    
    # Create subplot layout based on what we're plotting
    if plot_prob and plot_roc:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        prob_ax, roc_ax = axes
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if plot_prob:
            prob_ax = ax
        else:
            roc_ax = ax
    
    # Plot probability-based relational uncertainty
    if plot_prob:
        # Assuming model.prob_uncertainty_relation_history is a dict with keys as (stim1, stim2) tuples
        for pair, uncertainty_history in model.prob_uncertainty_relation_history.items():
            i, j = pair
            label = f"{stimulus_labels[i]} vs {stimulus_labels[j]}"
            prob_ax.plot(uncertainty_history, label=label)
        
        prob_ax.set_ylabel("Uncertainty (Probability)")
        prob_ax.set_title("Probability-Based Relational Uncertainty")
        prob_ax.grid(True, alpha=0.3)
        prob_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Plot ROC-based relational uncertainty
    if plot_roc:
        # Assuming model.roc_uncertainty_relation_history is a dict with keys as (stim1, stim2) tuples
        for pair, uncertainty_history in model.roc_uncertainty_relation_history.items():
            i, j = pair
            label = f"{stimulus_labels[i]} vs {stimulus_labels[j]}"
            roc_ax.plot(uncertainty_history, label=label)
        
        roc_ax.set_ylabel("Uncertainty (ROC)")
        roc_ax.set_title("ROC-Based Relational Uncertainty")
        roc_ax.grid(True, alpha=0.3)
        roc_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set common x-axis label
    if plot_prob and plot_roc:
        axes[-1].set_xlabel("Trial")
    else:
        plt.xlabel("Trial")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    
def plot_positions(model, stimulus_labels=None, save=None):
    """
    plot positions over trials
    
    Parameters:
        - model : Betasort
            -fitted model with position history
        - stimulus_labels : list, optional
            - labels for stimuli
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    plt.figure(figsize=(12, 8))
    position_array = np.array(model.position_history) # convert into array
    
    for i in range(model.n_stimuli):
        plt.plot(position_array[:, i], label=stimulus_labels[i])
        
    plt.xlabel("Trial")
    plt.ylabel("Estimated Position")
    plt.title("Estimated Stimulus Positions from Betasort Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
        
def plot_positions_across_days(all_models, mode='detailed', stimulus_labels=None, figsize=(12, 8), 
                              show_markers=False, save=None):
    """
    Plot positions of all stimuli across days.
    
    Parameters:
        - all_models : dict
            Dictionary where keys are day numbers and values are Betasort model instances
        - mode : str, optional
            'summary' (default): Plot only the final positions for each day
            'detailed': Plot the full evolution of positions across all trials and days
        - stimulus_labels : list, optional
            Labels for stimuli
        - figsize : tuple, optional
            Figure size
        - show_markers : bool, optional
            Whether to show markers at each trial point in detailed mode
        - save : str, optional
            Path to save the figure
            
    Returns:
        - fig, ax : matplotlib figure and axes objects
    """
    # Get sorted days
    days = sorted(all_models.keys())
    
    # Determine the maximum number of stimuli across all days
    max_stimuli = max(model.n_stimuli for model in all_models.values())
    
    # If no labels provided, create default ones
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(max_stimuli)]
    
    # Create a colormap to use different colors for each stimulus
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, max_stimuli))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if mode == 'summary':
        # Summary mode: plot only final positions for each day
        for i in range(max_stimuli):
            positions = []
            days_with_stimulus = []
            
            for day in days:
                model = all_models[day]
                if i < model.n_stimuli:
                    # Get the final position for this stimulus on this day
                    final_positions = model.position_history[-1]
                    positions.append(final_positions[i])
                    days_with_stimulus.append(day)
            
            if positions:  # Only plot if this stimulus appears in any day
                label = stimulus_labels[i] if i < len(stimulus_labels) else f"Stimulus {i}"
                ax.plot(days_with_stimulus, positions, 'o-', color=colors[i], label=label, markersize=8)
        
        # Add labels and title
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Estimated Position', fontsize=12)
        ax.set_title('Final Estimated Stimulus Positions by Day', fontsize=14)
        
        # Adjust x-ticks to show all days
        ax.set_xticks(days)
        ax.set_xticklabels([str(day) for day in days])
    
    elif mode == 'detailed':
        # Detailed mode: plot full evolution across all trials and days
        trial_offset = 0
        day_boundaries = [0]  # Start with 0 as the first boundary
        day_midpoints = []
        
        # Keep track of which stimuli we've already seen and labeled
        labeled_stimuli = set()
        
        # Plot positions for each stimulus across all days
        for day in days:
            model = all_models[day]
            n_trials = len(model.position_history)
            
            # Calculate midpoint for day label
            day_midpoints.append(trial_offset + n_trials / 2)
            
            # Get x-axis values for this day (trial numbers + offset)
            x_values = np.arange(trial_offset, trial_offset + n_trials)
            
            # Convert position_history to numpy array for easier indexing
            position_array = np.array(model.position_history)
            
            # Plot each stimulus for this day
            for i in range(model.n_stimuli):
                # Extract positions for this stimulus across all trials
                stimulus_positions = position_array[:, i]
                
                # Only add a label if we haven't seen this stimulus before
                if i not in labeled_stimuli:
                    label = stimulus_labels[i] if i < len(stimulus_labels) else f"Stimulus {i}"
                    labeled_stimuli.add(i)
                else:
                    label = None
                
                # Choose line style based on show_markers flag
                if show_markers:
                    line_style = 'o-'  # Line with circle markers
                    markersize = 4     # Smaller markers for detailed view
                else:
                    line_style = '-'   # Just a line
                    markersize = None  # No markers
                
                ax.plot(x_values, stimulus_positions, line_style, color=colors[i], 
                       label=label, markersize=markersize)
            
            # Update the trial offset for the next day
            trial_offset += n_trials
            day_boundaries.append(trial_offset)
        
        # Add vertical lines to separate days
        for boundary in day_boundaries[1:-1]:  # Skip the first and last boundaries
            ax.axvline(x=boundary, color='k', linestyle=':', alpha=0.5)
        
        # Add day labels at the top
        for i, day in enumerate(days):
            ax.text(day_midpoints[i], 1.02, f"Day {day}", ha='center', va='bottom', 
                   transform=ax.get_xaxis_transform())
        
        # Add labels and title
        ax.set_xlabel('Trial (continuous across days)', fontsize=12)
        ax.set_ylabel('Estimated Position', fontsize=12)
        ax.set_title('Evolution of Stimulus Positions Across All Trials and Days', fontsize=14)
        
        # Remove x-ticks for clarity (there would be too many trials)
        ax.set_xticks([])
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'summary' or 'detailed'.")
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend (outside the plot if there are many stimuli)
    if max_stimuli > 5:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    else:
        ax.legend(loc='best')
        plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, ax

def plot_uncertainty_across_days(all_models, uncertainty_type='ROC', mode='detailed', 
                                stimulus_labels=None, figsize=(12, 8), show_markers=False, save=None):
    """
    Plot uncertainty values of all stimulus pairs across days.
    
    Parameters:
        - all_models : dict
            Dictionary where keys are day numbers and values are Betasort model instances
        - uncertainty_type : str, optional
            'ROC' (default): Plot ROC-based uncertainty
            'relational': Plot probability-based relational uncertainty
            'stimulus': Plot individual stimulus uncertainty
        - mode : str, optional
            'summary' (default): Plot only the final uncertainty values for each day
            'detailed': Plot the full evolution of uncertainty across all trials and days
        - stimulus_labels : list, optional
            Labels for stimuli
        - figsize : tuple, optional
            Figure size
        - show_markers : bool, optional
            Whether to show markers at each trial point in detailed mode
        - save : str, optional
            Path to save the figure
            
    Returns:
        - fig, ax : matplotlib figure and axes objects
    """
    # Get sorted days
    days = sorted(all_models.keys())
    
    # Determine the maximum number of stimuli across all days
    max_stimuli = max(model.n_stimuli for model in all_models.values())
    
    # If no labels provided, create default ones
    if stimulus_labels is None:
        if uncertainty_type == 'stimulus':
            stimulus_labels = [f"Stimulus {i}" for i in range(max_stimuli)]
        else:
            # For pair-based uncertainties
            stimulus_labels = [f"Pair {i}-{i+1}" for i in range(max_stimuli-1)]
    
    # Create a colormap to use different colors for each stimulus/pair
    import matplotlib.cm as cm
    if uncertainty_type == 'stimulus':
        colors = cm.tab10(np.linspace(0, 1, max_stimuli))
        num_entities = max_stimuli
    else:
        colors = cm.tab10(np.linspace(0, 1, max_stimuli-1))
        num_entities = max_stimuli-1  # Number of pairs is one less than number of stimuli
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if mode == 'summary':
        # Summary mode: plot only final uncertainty values for each day
        for i in range(num_entities):
            values = []
            days_with_entity = []
            
            for day in days:
                model = all_models[day]
                
                # Skip if this stimulus/pair doesn't exist in this day's model
                if uncertainty_type == 'stimulus' and i >= model.n_stimuli:
                    continue
                elif uncertainty_type != 'stimulus' and i >= model.n_stimuli-1:
                    continue
                
                # Get the appropriate uncertainty history based on type
                if uncertainty_type == 'ROC':
                    history = model.ROC_uncertainty_history
                elif uncertainty_type == 'relational':
                    history = model.relational_uncertainty_history
                elif uncertainty_type == 'stimulus':
                    history = model.uncertainty_history
                else:
                    raise ValueError(f"Invalid uncertainty_type: {uncertainty_type}")
                
                # Get the final uncertainty value for this entity on this day
                final_values = history[-1]
                values.append(final_values[i])
                days_with_entity.append(day)
            
            if values:  # Only plot if this entity appears in any day
                if uncertainty_type == 'stimulus':
                    label = stimulus_labels[i] if i < len(stimulus_labels) else f"Stimulus {i}"
                else:
                    label = stimulus_labels[i] if i < len(stimulus_labels) else f"Pair {i}-{i+1}"
                
                ax.plot(days_with_entity, values, 'o-', color=colors[i], label=label, markersize=8)
        
        # Add labels and title
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Uncertainty', fontsize=12)
        
        if uncertainty_type == 'ROC':
            title = 'ROC Uncertainty Values by Day'
        elif uncertainty_type == 'relational':
            title = 'Relational Uncertainty Values by Day'
        else:
            title = 'Stimulus Uncertainty Values by Day'
            
        ax.set_title(title, fontsize=14)
        
        # Adjust x-ticks to show all days
        ax.set_xticks(days)
        ax.set_xticklabels([str(day) for day in days])
    
    elif mode == 'detailed':
        # Detailed mode: plot full evolution across all trials and days
        trial_offset = 0
        day_boundaries = [0]  # Start with 0 as the first boundary
        day_midpoints = []
        
        # Keep track of which entities we've already labeled
        labeled_entities = set()
        
        # Plot uncertainty values for each entity across all days
        for day in days:
            model = all_models[day]
            
            # Get the appropriate uncertainty history based on type
            if uncertainty_type == 'ROC':
                history = model.ROC_uncertainty_history
            elif uncertainty_type == 'relational':
                history = model.relational_uncertainty_history
            elif uncertainty_type == 'stimulus':
                history = model.uncertainty_history
            
            n_trials = len(history)
            
            # Calculate midpoint for day label
            day_midpoints.append(trial_offset + n_trials / 2)
            
            # Get x-axis values for this day (trial numbers + offset)
            x_values = np.arange(trial_offset, trial_offset + n_trials)
            
            # Convert history to numpy array for easier indexing
            uncertainty_array = np.array(history)
            
            # Plot each entity for this day
            if uncertainty_type == 'stimulus':
                n_entities = model.n_stimuli
            else:
                n_entities = model.n_stimuli - 1
                
            for i in range(n_entities):
                # Extract uncertainty values for this entity across all trials
                entity_values = uncertainty_array[:, i]
                
                # Only add a label if we haven't seen this entity before
                if i not in labeled_entities:
                    if uncertainty_type == 'stimulus':
                        label = stimulus_labels[i] if i < len(stimulus_labels) else f"Stimulus {i}"
                    else:
                        label = stimulus_labels[i] if i < len(stimulus_labels) else f"Pair {i}-{i+1}"
                    labeled_entities.add(i)
                else:
                    label = None
                
                # Choose line style based on show_markers flag
                if show_markers:
                    line_style = 'o-'  # Line with circle markers
                    markersize = 4     # Smaller markers for detailed view
                else:
                    line_style = '-'   # Just a line
                    markersize = None  # No markers
                
                ax.plot(x_values, entity_values, line_style, color=colors[i], 
                       label=label, markersize=markersize)
            
            # Update the trial offset for the next day
            trial_offset += n_trials
            day_boundaries.append(trial_offset)
        
        # Add vertical lines to separate days - now more dotted and visible
        for boundary in day_boundaries[1:-1]:  # Skip the first and last boundaries
            ax.axvline(x=boundary, color='k', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Add day labels at the top
        for i, day in enumerate(days):
            ax.text(day_midpoints[i], 1.02, f"Day {day}", ha='center', va='bottom', 
                   transform=ax.get_xaxis_transform())
        
        # Add labels and title
        ax.set_xlabel('Trial (continuous across days)', fontsize=12)
        ax.set_ylabel('Uncertainty', fontsize=12)
        
        if uncertainty_type == 'ROC':
            title = 'Evolution of ROC Uncertainty Values Across All Trials and Days'
        elif uncertainty_type == 'relational':
            title = 'Evolution of Relational Uncertainty Values Across All Trials and Days'
        else:
            title = 'Evolution of Stimulus Uncertainty Values Across All Trials and Days'
            
        ax.set_title(title, fontsize=14)
        
        # Remove x-ticks for clarity (there would be too many trials)
        ax.set_xticks([])
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'summary' or 'detailed'.")
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend (outside the plot if there are many entities)
    if num_entities > 5:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    else:
        ax.legend(loc='best')
        plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, ax
    
def plot_beta_distributions(model, x_resolution=1000, stimulus_labels=None, figsize=(12, 8), save=None):
    """
    plot beta distributions based on upper and lower parameters at the end
    
    Parameters:
        - model : Betasort
            - fitted model with U & L parameters
        - x_resolution : int, optional
            - higher gives smoother curves
        - stimulus_labels : list, optional
        - figsize : tuple, optional
    
    Returns:
        - fig, ax : matplotlib figure and axes objects
    """
    
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(0, 1, x_resolution)
    
    # maximum density value to scale the plot
    max_density = 0
    
    # for each stimulus
    for i in range(model.n_stimuli):
        # get parameters for beta distribution
        a = model.U[i] + 1
        b = model.L[i] + 1
        
        # create beta distribution
        beta_dist = stats.beta(a, b)
        
        # calculate probability density
        density = beta_dist.pdf(x)
        
        # track maximum density for scaling
        if max(density) > max_density:
            max_density = max(density)
        
        # plot
        ax.plot(x, density, label=f"{stimulus_labels[i]} (alpha={a:.2f}, beta={b:.2f})")
        
        # calculate and mark the mean or expected position
        mean_pos = a / (a + b)
        mean_density = beta_dist.pdf(mean_pos)
        ax.scatter([mean_pos], [mean_density], marker='o', s=50,
                   edgecolor='black', linewidth=1.5, zorder=5)
        
    # add shade
    for i in range(model.n_stimuli):
        a = model.U[i] + 1
        b = model.L[i] + 1
        beta_dist = stats.beta(a, b)
        density = beta_dist.pdf(x)
        ax.fill_between(x, density, alpha=0.1)
    
    # labels
    ax.set_xlabel("Position value", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Beta Distribution for Each Stimulus", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # legend
    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_boundaries_history(model, stimulus_labels=None, save=None):
    """
    plots thei hsitory of the upper and lower parameters
    
    Parameters:
        - model : Betasort
        - stimulus_labels : list, optional
            - labels of stimuli to plot, defaults to all
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    # plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # number of trials
    n_trials = len(model.U_history)
    trials = range(n_trials)
    
    for i, label in enumerate(stimulus_labels):
        ax.plot(trials, [u[i] for u in model.U_history],
                label=f"U - {label}", linestyle="-")
        ax.plot(trials, [l[i] for l in model.L_history],
                label=f"L - {label}", linestyle="--")

        # add final values annotation
        final_U = model.U_history[-1][i]
        final_L = model.L_history[-1][i]
        ax.annotate(f"U{i} = {final_U:.2f}", 
                    xy=(n_trials-1, final_U), xytext=(n_trials-10, final_U*1.05),
                    arrowprops=dict(arrowstyle="->"))
        ax.annotate(f"L{i} = {final_L:.2f}", 
                    xy=(n_trials-1, final_L), xytext=(n_trials-10, final_L*0.95),
                    arrowprops=dict(arrowstyle="->"))
    
    # labels
    ax.set_xlabel("Trial")
    ax.set_ylabel("Parameter Value")
    ax.set_title("History of U and L Parameters")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_match_rates(matches, window_size=10, save=None):
    """
    plot trial-by-trial and moving average match rates
    
    Parameters:
        - matches : array
            - trial by trial match rate
        - window_size : int
            - size of the moving average window
    """
    
    n_trials = len(matches)
    trials = np.arange(1, n_trials + 1)
    
    # calculate moving average
    moving_avg = np.convolve(matches, np.ones(window_size)/window_size, mode='valid')
    moving_avg_trials = np.arange(window_size, n_trials + 1)
    
    # plot
    plt.figure(figsize=(12, 6))
    plt.scatter(trials, matches, alpha=0.5, label="Trial-by-trial match rate")
    
    # plot moving average
    plt.plot(moving_avg_trials, moving_avg, 'r-', linewidth=2,
             label=f"Moving average (window={window_size})")

    # plot chance level
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Chance Level')
    
    # plot cumulative average
    cumulative_avg = np.cumsum(matches) / np.arange(1, len(matches) + 1)
    plt.plot(trials, cumulative_avg, 'g-', linewidth=2, label='Cumulative Average')
    
    # labels
    plt.xlabel("Trials")
    plt.ylabel("Match Rate")
    plt.title("Betasort Model Choices vs Real Rodent Choices")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def parameter_performance_heatmap(param_performances, save=None):
    plt.figure(figsize=(10, 8))
    xi_list = sorted(set(p[0] for p in param_performances.keys()))
    tau_list = sorted(set(p[1] for p in param_performances.keys()))
    
    # performance matrix for heatmap
    performance_matrix = np.zeros((len(xi_list), len(tau_list)))
    for i, xi in enumerate(xi_list):
        for j, tau in enumerate(tau_list):
            performance_matrix[i, j] = param_performances.get((xi, tau), 0)
    
    # plot heatmap
    plt.imshow(performance_matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Match Rate')
    plt.xlabel('Tau (noise parameter)')
    plt.ylabel('Xi (recall parameter)')
    plt.title('Parameter Performance Heatmap')
    
    # set tick labels
    plt.xticks(range(len(tau_list)), [f"{t:.3f}" for t in tau_list])
    plt.yticks(range(len(xi_list)), [f"{x:.3f}" for x in xi_list])
    
    # text annotations with values
    for i, xi in enumerate(xi_list):
        for j, tau in enumerate(tau_list):
            plt.text(j, i, f"{performance_matrix[i, j]:.3f}",
                     ha="center", va="center", color="w" if performance_matrix[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def parameter_performance_heatmap_with_threshold(param_performances, title=None, fixed_param=None, fixed_value=None, save=None):
    """
    Create a heatmap of parameter performances
    
    Parameters:
        - param_performances: Dictionary of performance metrics
        - title: Optional title for the plot
        - fixed_param: Which parameter to fix (0 for xi, 1 for tau, 2 for threshold)
        - fixed_value: The value to fix the parameter at
    
    Returns:
        - fig, ax: Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract parameter values
    params_structure = list(param_performances.keys())[0]
    
    if len(params_structure) == 3:  # (xi, tau, threshold)
        # We need to fix one parameter to visualize in 2D
        if fixed_param is None:
            fixed_param = 2  # Default to fixing threshold
            
        # Get unique values for each parameter
        xi_values = sorted(set(p[0] for p in param_performances.keys()))
        tau_values = sorted(set(p[1] for p in param_performances.keys()))
        threshold_values = sorted(set(p[2] for p in param_performances.keys()))
        
        if fixed_param == 0:  # Fix xi
            if fixed_value is None:
                fixed_value = xi_values[len(xi_values) // 2]  # Use middle value
            
            # Filter for fixed xi
            filtered_params = {(tau, thresh): perf 
                              for (xi, tau, thresh), perf in param_performances.items() 
                              if xi == fixed_value}
            
            # Use tau and threshold as axes
            x_label = 'Threshold'
            y_label = 'Tau (noise parameter)'
            x_values = threshold_values
            y_values = tau_values
            
            if title is None:
                title = f'Parameter Performance (Xi = {fixed_value:.3f})'
                
        elif fixed_param == 1:  # Fix tau
            if fixed_value is None:
                fixed_value = tau_values[len(tau_values) // 2]
                
            # Filter for fixed tau
            filtered_params = {(xi, thresh): perf 
                              for (xi, tau, thresh), perf in param_performances.items() 
                              if tau == fixed_value}
            
            # Use xi and threshold as axes
            x_label = 'Threshold'
            y_label = 'Xi (recall parameter)'
            x_values = threshold_values
            y_values = xi_values
            
            if title is None:
                title = f'Parameter Performance (Tau = {fixed_value:.3f})'
                
        else:  # Fix threshold (default)
            if fixed_value is None:
                fixed_value = threshold_values[len(threshold_values) // 2]
                
            # Filter for fixed threshold
            filtered_params = {(xi, tau): perf 
                              for (xi, tau, thresh), perf in param_performances.items() 
                              if thresh == fixed_value}
            
            # Use xi and tau as axes
            x_label = 'Tau (noise parameter)'
            y_label = 'Xi (recall parameter)'
            x_values = tau_values
            y_values = xi_values
            
            if title is None:
                title = f'Parameter Performance (Threshold = {fixed_value:.3f})'
    else:
        # Just (xi, tau) keys
        filtered_params = param_performances
        x_label = 'Tau (noise parameter)'
        y_label = 'Xi (recall parameter)'
        x_values = sorted(set(p[1] for p in param_performances.keys()))
        y_values = sorted(set(p[0] for p in param_performances.keys()))
        
        if title is None:
            title = 'Parameter Performance Heatmap'
    
    # Create performance matrix for heatmap
    performance_matrix = np.zeros((len(y_values), len(x_values)))
    
    # Fill the performance matrix
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            key = (y_val, x_val)
            if key in filtered_params:
                perf_value = filtered_params[key]
                if isinstance(perf_value, (list, np.ndarray)):
                    performance_matrix[i, j] = np.mean(perf_value)
                else:
                    performance_matrix[i, j] = perf_value
    
    # Plot heatmap
    im = ax.imshow(performance_matrix, interpolation='nearest', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Match Rate')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Set tick labels (limit number of ticks for readability)
    max_ticks = 10
    xtick_indices = np.linspace(0, len(x_values)-1, min(max_ticks, len(x_values))).astype(int)
    ytick_indices = np.linspace(0, len(y_values)-1, min(max_ticks, len(y_values))).astype(int)
    
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels([f"{x_values[i]:.3f}" for i in xtick_indices])
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels([f"{y_values[i]:.3f}" for i in ytick_indices])
    
    # Add text annotations if there aren't too many cells
    if len(y_values) * len(x_values) <= 100:
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                ax.text(j, i, f"{performance_matrix[i, j]:.3f}",
                        ha="center", va="center", 
                        color="w" if performance_matrix[i, j] < 0.7 else "black")
    
    return fig, ax

def plot_best_parameters(best_model):
    plot_positions(best_model)
    #plot_uncertainty(best_model)
    plot_beta_distributions(best_model)
    
def plot_vte_uncertainty(pair_vte_df, results, save=None):
    """
    Creates visualizations showing the relationship between VTE and pair-specific uncertainty
    
    Parameters:
        - pair_vte_df: DataFrame with VTE and uncertainty data
        - results: Analysis results from analyze_pair_specific_correlations
        - save: Directory to save plots (optional)
    """
    # 1. Boxplots of uncertainty by VTE for each pair
    unique_pairs = pair_vte_df['pair'].unique()
    
    # Determine plot grid dimensions
    n_pairs = len(unique_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    # For each uncertainty measure
    for measure in ['stim1_uncertainty', 'stim2_uncertainty', 'pair_relational_uncertainty', 'pair_roc_uncertainty']:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        fig.suptitle(f'{measure} by VTE Status for Each Stimulus Pair', fontsize=16)
        
        # Flatten axes array if necessary
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Create boxplot for each pair
        for i, pair in enumerate(unique_pairs):
            pair_data = pair_vte_df[pair_vte_df['pair'] == pair]
            
            if i < len(axes):
                if len(pair_data) > 0 and pair_data['vte_occurred'].nunique() > 1:
                    sns.boxplot(x='vte_occurred', y=measure, data=pair_data, ax=axes[i])
                    
                    # Add correlation info if available
                    if pair in results['by_pair'] and measure in results['by_pair'][pair]:
                        if isinstance(results['by_pair'][pair][measure], dict):
                            r = results['by_pair'][pair][measure]['r']
                            p = results['by_pair'][pair][measure]['p']
                            axes[i].set_title(f'Pair {pair}: r={r:.2f}, p={p:.3f}')
                        else:
                            axes[i].set_title(f'Pair {pair}')
                    else:
                        axes[i].set_title(f'Pair {pair}')
                else:
                    axes[i].set_visible(False)
                    
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for the suptitle
        
        if save:
            plt.savefig(os.path.join(save, f'vte_{measure}_by_pair.png'), dpi=300)
            plt.close()
        else:
            plt.show()
    
    # 2. Barplot showing correlation strength for each pair
    # Extract correlation values for each pair
    pairs = []
    correlation_values = []
    p_values = []
    measures = []
    
    for pair, pair_results in results['by_pair'].items():
        if isinstance(pair_results, dict):
            for measure in ['stim1_uncertainty', 'stim2_uncertainty', 'pair_relational_uncertainty', 'pair_roc_uncertainty']:
                if measure in pair_results and isinstance(pair_results[measure], dict):
                    pairs.append(pair)
                    correlation_values.append(pair_results[measure]['r'])
                    p_values.append(pair_results[measure]['p'])
                    measures.append(measure)
    
    if correlation_values:  # Only proceed if we have valid correlations
        # Create DataFrame for plotting
        corr_df = pd.DataFrame({
            'Pair': pairs,
            'Correlation': correlation_values,
            'P-value': p_values,
            'Measure': measures,
            'Significant': [p < 0.05 for p in p_values]
        })
        
        # Plot barplot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Pair', y='Correlation', hue='Measure', data=corr_df)
        
        # Add significance markers
        for i, (_, row) in enumerate(corr_df.iterrows()):
            if row['Significant']:
                plt.text(i, row['Correlation'] + 0.02, '*', ha='center', va='center', fontsize=12)
        
        plt.title('VTE-Uncertainty Correlation by Stimulus Pair')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(save, 'vte_correlation_by_pair.png'), dpi=300)
            plt.close()
        else:
            plt.show()
    
    # 3. Heatmap of uncertainty by day and pair for VTE trials
    # Calculate mean uncertainty for each day-pair combination
    heatmap_data = pair_vte_df.pivot_table(
        index='day', 
        columns='pair',
        values='pair_relational_uncertainty',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f")
    plt.title('Mean Relational Uncertainty by Day and Stimulus Pair')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(save, 'uncertainty_heatmap_by_day_pair.png'), dpi=300)
        plt.close()
    else:
        plt.show()
    
    # 4. Line plot of VTE occurrence rate and mean uncertainty over days
    day_summary = pair_vte_df.groupby('day').agg({
        'vte_occurred': 'mean',
        'stim1_uncertainty': 'mean',
        'stim2_uncertainty': 'mean',
        'pair_relational_uncertainty': 'mean',
        'pair_roc_uncertainty': 'mean'
    }).reset_index()
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Day')
    ax1.set_ylabel('VTE Rate', color=color)
    ax1.plot(day_summary['day'], day_summary['vte_occurred'], 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Uncertainty', color=color)
    ax2.plot(day_summary['day'], day_summary['pair_relational_uncertainty'], 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('VTE Rate and Mean Uncertainty Over Days')
    fig.tight_layout()
    
    if save:
        plt.savefig(os.path.join(save, 'vte_uncertainty_over_days.png'), dpi=300)
        plt.close()
    else:
        plt.show()
    
    

## CENTRAL FUNCTION THAT CONTAINS EVERYTHING FOR THE PIPELINE
def analyze_betasort_comprehensive(all_data_df, rat, n_simulations=100, use_diff_evolution=True,
                                  xi=None, tau=None, threshold=None):
    """
    Comprehensive analysis function for Betasort model that:
    1. Finds optimal parameters (optionally using differential evolution)
    2. Runs a model with optimal parameters
    3. Performs binomial tests, t-tests, and VTE uncertainty analysis
    
    Parameters:
    - all_data_df: DataFrame containing rat choice data
    - rat: String identifier for the rat
    - n_simulations: Number of simulations for choice testing
    - use_diff_evolution: Whether to find optimal parameters using differential evolution
    - xi, tau, threshold: Optional parameter values to use if not using differential evolution
    
    Returns:
    - Dictionary containing:
        - all_models: Dictionary of models for each day
        - pair_vte_df: DataFrame with paired VTE and uncertainty values
        - best_xi: Optimal xi parameter
        - best_tau: Optimal tau parameter
        - best_threshold: Optimal threshold parameter
        - best_performance: Performance with optimal parameters
        - session_results_ttest: T-test results by session
        - overall_results_ttest: Overall t-test results
        - session_results_binomial: Binomial test results by session
    """
    results = {}
    
    # Step 1: Find optimal parameters if requested
    if use_diff_evolution:
        print("Finding optimal parameters using differential evolution...")
        best_xi, best_tau, best_threshold, best_performance = diff_evolution_lite(all_data_df, rat)
    else:
        # Use provided parameters
        best_xi = xi if xi is not None else 0.95  # Default values
        best_tau = tau if tau is not None else 0.05
        best_threshold = threshold if threshold is not None else 0.8
    
    # Store parameter values
    results['best_xi'] = best_xi
    results['best_tau'] = best_tau
    results['best_threshold'] = best_threshold
    
    print(f"Using parameters: xi={best_xi}, tau={best_tau}, threshold={best_threshold}")
    
    # Step 2: Initialize storage for all analyses
    all_models = {}
    global_U, global_L, global_R, global_N = {}, {}, {}, {}
    
    # VTE data collection
    pair_vte_data = []
    
    # Binomial analysis storage
    session_results_binomial = {}
    
    # T-test analysis storage
    session_results_regression = []
    
    # Match rates for performance calculation
    all_match_rates = []
    
    # Process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        print(f"Processing day {day}...")
        
        # Extract relevant data for this day
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        rewards = day_data["correct"].values
        
        # Handle optional VTE column
        if 'VTE' in day_data.columns:
            vtes = day_data["VTE"].values
        else:
            vtes = np.zeros_like(chosen_idx)  # Default to no VTEs if not provided
            
        # Handle optional ID column
        if 'ID' in day_data.columns:
            traj_nums = day_data["ID"].values
        else:
            traj_nums = np.arange(len(chosen_idx))  # Default to sequential IDs
        
        # Identify which stimuli are present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1  # +1 because of 0-indexing
        
        # Initialize a model for this day
        model = Betasort(n_stimuli, rat, day, tau=best_tau, xi=best_xi)
        
        # Transfer state from previous days
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
        
        # Reset histories
        model.uncertainty_history = [model.get_all_stimulus_uncertainties()]
        model.relational_uncertainty_history = [model.get_all_relational_uncertainties()]
        model.ROC_uncertainty_history = [model.get_all_ROC_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.U_history = [model.U.copy()]
        model.L_history = [model.L.copy()]
        
        # Data for analyses
        day_matches = []
        model_correct_rates = []
        rat_correct_rates = []
        
        # Process trials for this day
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            vte = vtes[t]
            traj_num = traj_nums[t]
            
            # VTE Analysis: Get uncertainties before updates
            stim1_uncertainty = model.get_uncertainty_stimulus(min(chosen, unchosen))
            stim2_uncertainty = model.get_uncertainty_stimulus(max(chosen, unchosen))
            pair_relational_uncertainty = model.get_uncertainty_relation(min(chosen, unchosen), max(chosen, unchosen))
            pair_roc_uncertainty = model.get_uncertainty_relation_ROC(min(chosen, unchosen), max(chosen, unchosen))
            
            # Store VTE data
            vte_occurred = 1 if vte else 0
            pair_vte_data.append({
                'day': day,
                'trial_num': traj_num,
                'stim1': min(chosen, unchosen),
                'stim2': max(chosen, unchosen),
                'chosen': chosen,
                'unchosen': unchosen,
                'vte_occurred': vte_occurred,
                'stim1_uncertainty': stim1_uncertainty,
                'stim2_uncertainty': stim2_uncertainty,
                'pair_relational_uncertainty': pair_relational_uncertainty,
                'pair_roc_uncertainty': pair_roc_uncertainty,
                'reward': reward
            })
            
            # Simulate model choices (for all analyses)
            model_choices = np.zeros(n_simulations)
            model_correct = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen, unchosen])
                model_choices[sim] = model_choice
                # Correct = choosing the lower-valued stimulus (as per original code)
                model_correct[sim] = 1 if model_choice == min(chosen, unchosen) else 0
            
            # Calculate match rate (model choice matches rat's choice)
            model_match_rate = np.mean(model_choices == chosen)
            day_matches.append(model_match_rate)
            
            # Store correct rates for binomial analysis
            model_correct_rates.append(np.mean(model_correct))
            rat_correct_rates.append(1 if chosen < unchosen else 0)
            
            # Update model based on actual choice
            model.thresholding_update(chosen, unchosen, reward, model_match_rate, threshold=best_threshold)
        
        # Store the model for this day
        all_models[day] = model
        
        # Update global states for the next day
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
        
        # Calculate day-specific match rate for performance
        day_match_rate = np.mean(day_matches)
        all_match_rates.append(day_match_rate)
        
        # Binomial analysis for this day
        n_rat_correct = int(np.sum(rat_correct_rates))
        n_trials = len(rat_correct_rates)
        model_correct_rate = sum(model_correct_rates) / len(model_correct_rates)
        p_value_binomial = stats.binomtest(n_rat_correct, n_trials, p=model_correct_rate)
        
        session_results_binomial[day] = {
            'matches': n_rat_correct,
            'trials': n_trials,
            'match_rate': n_rat_correct/n_trials if n_trials > 0 else 0,
            'p_value': p_value_binomial,
            'model_rate': model_correct_rate,
            'significant': p_value_binomial.pvalue < 0.05
        }
        
        # calculate logistic regression
        X = np.array(model_correct_rates).reshape(-1, 1) # model probabilities as features
        Y = np.array(rat_correct_rates)
        
        # Check if there's more than one class before fitting
        unique_classes = np.unique(Y)
        if len(unique_classes) > 1:
            regression_model = LogisticRegression()
            regression_model.fit(X, Y)
            
            # get model accuracy for logistic regression
            predictions = regression_model.predict(X)
            accuracy = accuracy_score(Y, predictions)
        else:
            # Handle single-class case
            # If all predictions are the same, the accuracy is either 0% or 100%
            accuracy = float(unique_classes[0])  # If all Y are 1, accuracy is 1.0; if all Y are 0, accuracy is 0.0
            
        session_results_regression.append(accuracy)
    
    # Calculate overall performance
    best_performance = np.mean(all_match_rates)
    results['best_performance'] = best_performance
    
    # Convert VTE data to DataFrame
    pair_vte_df = pd.DataFrame(pair_vte_data)
    if len(pair_vte_df) > 0:  # Add pair column for VTE analysis
        pair_vte_df['pair'] = pair_vte_df.apply(lambda row: f"{row['stim1']}-{row['stim2']}", axis=1)
    
    # Store all results
    results['all_models'] = all_models
    results['pair_vte_df'] = pair_vte_df
    results['session_results_binomial'] = session_results_binomial
    results['session_predictions_regression'] = session_results_regression
    
    return results