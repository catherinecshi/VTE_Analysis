import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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

        # initialise memory arrays
        self.U = np.ones(n_stimuli) # upper parameter
        self.L = np.ones(n_stimuli) # lower parameter
        self.R = np.ones(n_stimuli) # rewards
        self.N = np.ones(n_stimuli) # non rewards
        
        # store history for later plotting
        self.uncertainty_history = [self.get_all_stimulus_uncertainties()]
        self.relational_uncertainty_history = [self.get_all_relational_uncertainties()]
        self.ROC_uncertainty_history = [self.get_all_ROC_uncertainties()]
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
            #print(a, b, "a + b < 2")
            #print(self.rat, self.day, self.trial)
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
            auc += np.mean(samples1[i] > samples2)
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

    def update(self, chosen, unchosen, reward):
        """
        implements first updating policy
        
        Parameters:
            - chosen : int
                - index of chosen stimulus
            - unchosen : int
                - index of unchosen stimulus
            - reward : int
                - 1 for reward and 0 otherwise
            - threshold : float
                - the threshold for which the the model simulation rate of likely choice must be above
                  for consolidation to happen
        """
        # update trial count
        self.trial += 1
        #print(self.rat, self.day, self.trial)
        #print("before", self.U, self.L, chosen, unchosen, reward)
        #print(self.R, self.N)
        
        # relax
        self.R = self.R * self.xi
        self.N = self.N * self.xi
        
        # estimate trial reward rate
        E = self.R / (self.R + self.N)
        xi_R = E / (E + 1) + 0.5
        
        # relax some more :)
        self.U = self.U * xi_R * self.xi
        self.L = self.L * xi_R * self.xi

        # estimate sitmulus positions
        V = self.U / (self.U + self.L)

        if reward == 1:
            # update reward rate
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            # update both trial and inferred stimuli position
            # essentially, this configuration led to the correct choice so let's keep doing it
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

        #print("after", self.U, self.L)
        
        # store updated uncertainties and positions upper and lower
        self.uncertainty_history.append(self.get_all_stimulus_uncertainties())
        self.position_history.append(self.get_all_positions())
        self.U_history.append(self.U.copy())
        self.L_history.append(self.L.copy())
    
    def real_data_update(self, chosen, unchosen, reward, probability):
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
        non_likelihood = 1 - probability

        if reward == 1:
            # update reward rate
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            # update both trial and inferred stimuli position
            # essentially, this configuration led to the correct choice so let's keep doing it
            self.U = self.U + V
            self.L = self.L + (1 - V)
            
            # radical new moves
            self.U[chosen] = self.U[chosen] + V[chosen] # * probability
            #self.U[unchosen] = self.U[unchosen] + V[unchosen] * non_likelihood
            #self.L[chosen] = self.L[chosen] + (1 - V[chosen]) * probability
            self.L[unchosen] = self.L[unchosen] + (1 - V[unchosen]) # * non_likelihood
            
            # radical new moves 2
            # self.U = self.U + V * probability
            # self.L = self.L + (1 - V) * probability
        # elif reward == 1: # change above to make sure that it says reward == 1 and probability < 0.5 or arbitrary threshold - maybe check?
            # self.R[unchosen] = self.R[unchosen] + 1
            # self.R[chosen] = self.R[chosen] + 1
            
            # self.U = self.U + V
            # self.L = self.L + (1 - V)
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

        #print("after", self.U, self.L)
        
        # store updated uncertainties and positions upper and lower
        self.uncertainty_history.append(self.get_all_stimulus_uncertainties())
        self.position_history.append(self.get_all_positions())
        self.U_history.append(self.U.copy())
        self.L_history.append(self.L.copy())
        
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
        non_likelihood = 1 - probability

        if reward == 1 and probability < threshold:
            # update reward rate
            self.R[unchosen] = self.R[unchosen] + 1
            self.R[chosen] = self.R[chosen] + 1
            
            # update both trial and inferred stimuli position
            # essentially, this configuration led to the correct choice so let's keep doing it
            #self.U = self.U + V
            #self.L = self.L + (1 - V)
            
            # radical new moves
            self.U[chosen] = self.U[chosen] + probability
            #self.U[unchosen] = self.U[unchosen] + V[unchosen] * non_likelihood
            #self.L[chosen] = self.L[chosen] + (1 - V[chosen]) * probability
            self.L[unchosen] = self.L[unchosen] + probability
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

        #print("after", self.U, self.L)
        
        # store updated uncertainties and positions upper and lower
        self.uncertainty_history.append(self.get_all_stimulus_uncertainties())
        self.relational_uncertainty_history.append(self.get_all_relational_uncertainties())
        self.ROC_uncertainty_history.append(self.get_all_ROC_uncertainties())
        self.position_history.append(self.get_all_positions())
        self.U_history.append(self.U.copy())
        self.L_history.append(self.L.copy())
    
    def simulate_trials(self, chosen_idx, other_idx, n_simulations=100):
        """
        simulate n_simulations trials to get a rate at which one element would be picked over another
        
        Parameters:
            - chosen_idx : int
                - the index for the chosen element
            - other_idx : int
                - the index for the other element
            n_simulations : int, defaults to 100
                - the number of simulations ran for getting rate
        
        Returns:
            - model_match_rate : float
                - the rate at which the model produced choices equivalent to the actual choice
        """
        
        # run multiple simulations to get choice probability
        model_choices = np.zeros(n_simulations)
        for sim in range(n_simulations):
            model_choice = self.choose([chosen_idx, other_idx])
            model_choices[sim] = model_choice
        
        model_match_rate = np.mean(model_choices == chosen_idx)
        return model_match_rate
    
    def simulate_trial(self, stim_pair):
        """
        simulate a single trial with pair of stimuli
        
        Parameters:
            - stim_pair (list): two element list with indices of stimuli in the pair
        
        Returns:
            - chosen (int): index of chosen stimulus
            - reward (int): 1 if choice was rewarded, 0 otherwise
        """
        # choose one stimulus
        chosen_idx = self.choose(stim_pair)
        chosen = stim_pair.index(chosen_idx)
        unchosen = 1 - chosen
        unchosen_idx = stim_pair[unchosen]
        
        # determine whether rewards should be given and updates accordingly
        reward = 1 if chosen_idx < unchosen_idx else 0 # since earlier items are rewarded
        self.update(chosen_idx, unchosen_idx, reward)
        
        return chosen_idx, reward
    
def analyze_one_rat(all_data_df, rat, tau=0.05, xi=0.95, threshold=0.5):
    """
    uses real data from participants to update values for stimulus position and uncertainty
    takes all data from one rat and separately updates the model as elements get added
    
    Parameters:
        - all_data_df : DataFrame
            - data for all days for a single rat
        - rat : String
            - the rat
        - tau : float
            - noise parameter
        - xi : float
            - recall parameter
    
    Returns:
        - final_model : Betasort
            - final betasort model for all days
        - all_models : Dict
            - dictionary of models for each day
    """
    # sort data by day
    all_data_df = all_data_df.sort_values('Day')
    
    all_models = {}
    
    # track global stimuli states
    global_U = {}
    global_L = {}
    global_R = {}
    global_N = {}
    
    # process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        # Extract relevant data
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        rewards = day_data["correct"].values
        
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
        
        for t in range(n_trials):
            chosen_idx, unchosen_idx = participant_choices[t]
            reward = rewards[t]
            
            # Validate indices (just in case)
            if not (0 <= chosen_idx < n_stimuli) or not (0 <= unchosen_idx < n_stimuli):
                print(f"Day {day}, Trial {t}: Invalid indices - chosen {chosen_idx} unchosen {unchosen_idx}")
                continue
                
            # Update model
            model.update(chosen_idx, unchosen_idx, reward)
        
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
    return all_models[final_day], all_models

def analyze_real_data(participant_choices, feedback, n_stimuli, rat, day, tau=0.05, xi=0.95):
    """
    uses real data from participants to update values for stimulus position and uncertainty
    
    Parameters:
        - participant_choices : array-like, shape (n_trials, 2)
            - each row contains [chosen_idx, unchosen_idx] for each trial
        - feedback : array-like, n_trials long
            - binary array of feedback (1 for reward and 0 for none) for each trial
        - n_stimuli : int
            - number of stimuli in sequence
        - tau : float
            - noise parameter
        - xi : float
            - recall parameter controlling memory decay
    
    Returns:
        - model : Betasort
            - fitted model with uncertainty history
    """
    
    model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
    n_trials = len(participant_choices)
    
    # run through each trial
    for t in range(n_trials):
        chosen_idx, other_idx = participant_choices[t]
        reward = feedback[t]
        
        if not (0 <= chosen_idx < n_stimuli) or not (0 <= other_idx < n_stimuli):
            print(f"Trial {t}: Invalid indicies - chosen {chosen_idx} unchosen {other_idx}")
        
        # update model based on actual choice and feedback
        model.update(chosen_idx, other_idx, reward)
        
    return model

def compare_model_to_one_rat(all_data_df, rat, n_simulations=100, tau=0.01, xi=0.99, threshold=0.5):
    # sort data by day
    all_data_df = all_data_df.sort_values('Day')
    
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
            #model.update(chosen_idx, other_idx, reward)
            model.thresholding_update(chosen_idx, other_idx, reward, model_match_rate, threshold=threshold)
        
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

def compare_model_to_rats(participant_choices, n_stimuli, rat, day, tau=0.05, xi=0.95, n_simulations=100):
    """
    compares real choices to model predictions
    
    Parameters:
        - participant_choices : array-like, shape (n_trials, 2)
            - each row contians [chosen_idx, unchosen_idx] for each trial
        - n_stimuli: int
            - number of stimuli in sequence
        - tau : float
            - noise parameter
        - xi : float
            - recall parameter controlling memory decay
        - n_simulations : int
            - number of simulations to run per trial for model predictions
    
    Returns:
        - match_rates : array
            - trial-by-trial match rates between model and human choices
        - cumulative_match_rate : float
            - overall match rates across all trials
        - model : Betasort
            - the model after the final update
    """
    
    model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
    n_trials = len(participant_choices)
    
    # track matches between model and human
    matches = np.zeros(n_trials)
    
    # for each trial
    for t in range(n_trials):
        chosen_idx, other_idx = participant_choices[t]
        
        # check if the indices are valid
        if not (0 <= chosen_idx < n_stimuli) or not (0 <= other_idx < n_stimuli):
            print(f"Trial {t}: Invalid indices - chosen {chosen_idx} unchosen {other_idx}")
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
        model.update(chosen_idx, other_idx, reward)
    
    # calculate cumulative match rate
    cumulative_match_rate = np.mean(matches)
    
    return matches, cumulative_match_rate, model

def binomial_analysis_by_session(all_data_df, rat, tau=0.05, xi=0.95, n_simulations=100):
    """Run binomial test on shtuff"""
    
    all_data_df = all_data_df.sort_values('Day')
    
    session_results = {}
    for day, day_data in all_data_df.groupby('Day'):
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        # initialize model with current parameters
        model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        model_correct = np.zeros(len(chosen_idx))
        rat_correct = np.zeros(len(chosen_idx))
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
            
            model_correct[t] = np.mean(sim_correct)
            rat_correct[t] = reward
            model.thresholding_update(chosen_idx[t], unchosen_idx[t], reward, current_model_match_rate, threshold=0.6)
        
        # binomial test
        n_matches = int(np.sum(rat_correct))
        n_trials = len(rat_correct)
        model_correct_rate = np.mean(model_correct)
        p_value = stats.binomtest(n_matches, n_trials, p=model_correct_rate)
        
        session_results[day] = {
            'matches': n_matches,
            'trials': n_trials,
            'match_rate': n_matches/n_trials,
            'p_value': p_value,
            'model_rate': model_correct_rate,
            'significant': p_value.pvalue < 0.05
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
            
            # Store choices (as binary: 1 if model matches rat, 0 if different)
            match = 1 if model_choice == real_choice else 0
            
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

def find_optimal_parameters(participant_data, n_stimuli, rat, day, n_simulations=100, xi_values=None, tau_values=None):
    """
    Find the optimal parameter values to fit model onto the data
    
    Parameters:
        - participant_data : array-like
            - array of [chosen_idx, unchosen_idx] pairs
        - feedback : array-like
            - array of 1s and 0s depending on reward or not
        - n_stimuli : int
            - number of stimuli in the hierarhcy
        - rat, day : identifiers
        - n_simulations : int
            - number of simulations for each model prediction
        - xi_values, tau_values : array-like, defaults to None
            - parameter values to test
        
    Returns:
        - best_params : dict
            - best parameters found
        - param_performances : dict
            - performances on each parameter combination
    """
    
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
            _, matches_rate, _ = compare_model_to_rats(participant_data, n_stimuli, rat, day, tau=tau, xi=xi, 
                                                       n_simulations=n_simulations)
            
            param_performances[(xi, tau)] = matches_rate # store rate
    
    # find best parameter combination
    best_params = max(param_performances, key=param_performances.get)
    best_xi, best_tau = best_params
    best_performance = param_performances[best_params]
    
    return best_xi, best_tau, best_performance, param_performances

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
    # unpack parameters
    xi, tau, threshold = params
    all_data, rat = args
    
    # evaluate model with these parameters
    _, _, match_rates = compare_model_to_one_rat(
        all_data, rat,
        tau=tau, xi=xi, threshold=threshold
    )
    
    # return negative match rate since we're minimizing
    return -np.mean(match_rates)

def minimize_parameters(all_data, rat):
    #initial_params = [0.95, 0.05, 0.5] # xi, tau, threhsold
    
    result = differential_evolution(
        model_performance_objective,
        bounds=[(0.75, 1.0), (0.001, 0.25), (0.3, 0.7)],
        args=(all_data, rat),
        popsize=15,            # Population size (default is 15x number of parameters)
        mutation=(0.5, 1.0),   # Range for mutation scaling factor
        recombination=0.7,     # Probability of recombination for each parameter
        disp=True              # Show progress
    )
    
    best_xi, best_tau, best_threshold = result.x
    best_performance = -result.fun
    
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



def plot_stimulus_uncertainty(model, stimulus_labels=None):
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
    plt.show()
    
def plot_relational_uncertainty(model, stimulus_labels=None):
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
    plt.show()

def plot_ROC_uncertainty(model, stimulus_labels=None):
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
    plt.show()

def plot_relational_uncertainty_history(model, uncertainty_type='both', stimulus_labels=None, figsize=(15, 10)):
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
    plt.show()
    
def plot_positions(model, stimulus_labels=None):
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
    plt.show()
    
def plot_beta_distributions(model, x_resolution=1000, stimulus_labels=None, figsize=(12, 8)):
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
    plt.show()

def plot_boundaries_history(model, stimulus_labels=None):
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
    plt.show()

def plot_match_rates(matches, window_size=10):
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
    plt.show()

def parameter_performance_heatmap(param_performances):
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
    plt.show()

def parameter_performance_heatmap_with_threshold(param_performances, title=None, fixed_param=None, fixed_value=None):
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
    
    plot_stimulus_uncertainty()
    return fig, ax

def plot_best_parameters(best_model):
    plot_positions(best_model)
    plot_uncertainty(best_model)
    plot_beta_distributions(best_model)