import numpy as np

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