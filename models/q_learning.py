import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class QLearning:
    def __init__(self, n_stimuli, rat, day, alpha=0.1, gamma=0.95, tau=0.05, xi=0.95):
        """
        Initialize Q-learning model for transitive inference
        
        Parameters:
            - n_stimuli (int): number of stimuli in the list
            - rat (str): identifier for the rat
            - day (int): day of experiment
            - alpha (float): learning rate
            - gamma (float): discount factor
            - tau (float): temperature parameter for softmax (exploration)
            - xi (float): memory decay parameter (similar to Betasort)
        """
        self.n_stimuli = n_stimuli
        self.rat = rat
        self.day = day
        self.trial = 0
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.xi = xi
        
        # Unexpected uncertainty parameters
        self.seen_stimuli = set()
        self.unexpected_uncertainty = 0
        self.unexpected_uncertainty_decay = 0.9
        
        # Initialize Q-values (relative value of each stimulus)
        self.Q = np.zeros(n_stimuli)
        
        # Initialize reward/non-reward tracking (similar to Betasort)
        self.R = np.ones(n_stimuli)  # rewards
        self.N = np.ones(n_stimuli)  # non-rewards
        
        # Initialize uncertainty estimates
        self.uncertainty = np.ones(n_stimuli)
        
        # Store history for analysis
        self.uncertainty_history = [self.get_all_uncertainties()]
        self.relational_uncertainty_history = [self.get_all_relational_uncertainties()]
        self.position_history = [self.get_all_positions()]
        self.Q_history = [self.Q.copy()]
        self.unexpected_uncertainty_history = [0]
    
    def choose(self, available_stimuli):
        """
        Choice policy using softmax with uncertainty-weighted exploration
        
        Parameters:
            - available_stimuli (list): indices of available stimuli to choose from
            
        Returns:
            - chosen stimulus index
        """
        if np.random.random() < self.tau:
            # Random exploration with probability tau
            return np.random.choice(available_stimuli)
        
        # Get Q-values for available stimuli
        q_values = np.array([self.Q[i] for i in available_stimuli])
        
        # Apply softmax for action selection
        # Higher temperature = more exploration, lower = more exploitation
        temperature = 0.5  # Can be adjusted
        
        # Add small constant to prevent overflow in exp
        q_values = q_values - np.max(q_values)
        
        # Calculate softmax probabilities
        exp_q = np.exp(q_values / temperature)
        probabilities = exp_q / np.sum(exp_q)
        
        # Choose based on probability distribution
        if len(available_stimuli) > 1:
            chosen_idx = np.random.choice(len(available_stimuli), p=probabilities)
            return available_stimuli[chosen_idx]
        else:
            return available_stimuli[0]
    
    def get_uncertainty_stimulus(self, stimulus_idx):
        """
        Get uncertainty estimate for a stimulus
        
        Parameters:
            - stimulus_idx (int): index of the stimulus
            
        Returns:
            - (float): uncertainty value between 0 and 1
        """
        return self.uncertainty[stimulus_idx]
    
    def get_all_uncertainties(self):
        """Get uncertainty values for all stimuli"""
        return self.uncertainty.copy()
    
    def get_uncertainty_relation(self, stim1_idx, stim2_idx):
        """
        Calculate uncertainty about the relationship between two stimuli
        
        Parameters:
            - stim1_idx, stim2_idx (int): indices of stimuli to compare
            
        Returns:
            - (float): uncertainty value between 0 and 1
        """
        # Uncertainty is higher when Q-values are close
        q_diff = abs(self.Q[stim1_idx] - self.Q[stim2_idx])
        # Transform to 0-1 scale (higher = more uncertain)
        uncertainty = np.exp(-q_diff)
        return uncertainty
    
    def get_all_relational_uncertainties(self):
        """Get uncertainties for all adjacent pairs of stimuli"""
        return np.array([self.get_uncertainty_relation(i, i+1) for i in range(self.n_stimuli-1)])
    
    def get_all_positions(self):
        """Get normalized positions (Q-values scaled to 0-1) for all stimuli"""
        if np.max(self.Q) - np.min(self.Q) == 0:
            return np.linspace(0, 1, self.n_stimuli)
        
        # Scale Q-values to 0-1 range
        scaled_q = (self.Q - np.min(self.Q)) / (np.max(self.Q) - np.min(self.Q))
        return scaled_q
    
    def update(self, chosen, unchosen, reward, probability, threshold=0.75):
        """
        Q-learning update with memory decay and implicit inference
        
        Parameters:
            - chosen (int): index of chosen stimulus
            - unchosen (int): index of unchosen stimulus
            - reward (int): 1 for reward, 0 for no reward
            - probability (float): model's confidence in the choice
            - threshold (float): threshold for confidence-based updating
        """
        # Update trial count
        self.trial += 1
        
        # Update unexpected uncertainty based on new stimuli
        new_stimuli_encountered = False
        if chosen not in self.seen_stimuli:
            self.seen_stimuli.add(chosen)
            new_stimuli_encountered = True
        if unchosen not in self.seen_stimuli:
            self.seen_stimuli.add(unchosen)
            new_stimuli_encountered = True
        
        if new_stimuli_encountered:
            self.unexpected_uncertainty = 1.0
        else:
            self.unexpected_uncertainty *= self.unexpected_uncertainty_decay
        
        # Calculate effective memory decay based on unexpected uncertainty
        effective_xi = 1 - 0.5 * self.unexpected_uncertainty
        
        # Memory decay step (similar to Betasort's relax)
        self.R = self.R * effective_xi
        self.N = self.N * effective_xi
        self.Q = self.Q * effective_xi
        
        # Update reward tracking
        if reward == 1:
            self.R[chosen] = self.R[chosen] + 1
            self.R[unchosen] = self.R[unchosen] + 1
        else:
            self.N[chosen] = self.N[chosen] + 1
            self.N[unchosen] = self.N[unchosen] + 1
        
        # Compute target Q-values
        if reward == 1:
            # If correct (reward), the chosen stimulus should have a higher value than unchosen
            target_diff = 1.0  # Target difference in Q-values
        else:
            # If incorrect (no reward), the unchosen stimulus should have a higher value
            target_diff = -1.0
        
        # Calculate learning rate modulation based on uncertainty
        alpha_chosen = self.alpha * (1 + self.uncertainty[chosen])
        alpha_unchosen = self.alpha * (1 + self.uncertainty[unchosen])
        
        # More aggressive updating when model confidence is low (similar to Betasort)
        if probability < threshold:
            alpha_chosen *= 2
            alpha_unchosen *= 2
        
        # Standard Q-learning update for chosen and unchosen stimuli
        if reward == 1:
            # Increase value of chosen stimulus
            self.Q[chosen] += alpha_chosen * (target_diff - (self.Q[chosen] - self.Q[unchosen]))
            # Decrease value of unchosen stimulus 
            self.Q[unchosen] -= alpha_unchosen * (target_diff - (self.Q[chosen] - self.Q[unchosen]))
        else:
            # Decrease value of chosen stimulus
            self.Q[chosen] += alpha_chosen * (target_diff - (self.Q[chosen] - self.Q[unchosen]))
            # Increase value of unchosen stimulus
            self.Q[unchosen] -= alpha_unchosen * (target_diff - (self.Q[chosen] - self.Q[unchosen]))
        
        # Implicit inference for other stimuli (similar to Betasort)
        # Update values of stimuli that fall between chosen and unchosen
        for j in range(self.n_stimuli):
            if j != chosen and j != unchosen:
                # Skip stimuli not involved in current trial
                if not (min(self.Q[chosen], self.Q[unchosen]) <= self.Q[j] <= max(self.Q[chosen], self.Q[unchosen])):
                    continue
                
                # Determine relative position between chosen and unchosen
                if abs(self.Q[chosen] - self.Q[unchosen]) > 0:
                    relative_pos = (self.Q[j] - min(self.Q[chosen], self.Q[unchosen])) / abs(self.Q[chosen] - self.Q[unchosen])
                else:
                    relative_pos = 0.5
                
                # Implicit update based on relative position
                alpha_implicit = self.alpha * 0.5  # Smaller learning rate for implicit updates
                
                if reward == 1:
                    if self.Q[chosen] > self.Q[unchosen]:
                        # Interpolate between chosen (higher) and unchosen (lower)
                        target = self.Q[unchosen] + relative_pos * (self.Q[chosen] - self.Q[unchosen])
                    else:
                        # Interpolate between unchosen (higher) and chosen (lower)
                        target = self.Q[chosen] + relative_pos * (self.Q[unchosen] - self.Q[chosen])
                else:
                    if self.Q[chosen] < self.Q[unchosen]:
                        # Interpolate between chosen (lower) and unchosen (higher)
                        target = self.Q[unchosen] - relative_pos * (self.Q[unchosen] - self.Q[chosen])
                    else:
                        # Interpolate between unchosen (lower) and chosen (higher)
                        target = self.Q[chosen] - relative_pos * (self.Q[chosen] - self.Q[unchosen])
                
                # Update Q-value based on implicit inference
                self.Q[j] += alpha_implicit * (target - self.Q[j])
        
        # Update uncertainty estimates
        self.update_uncertainties(chosen, unchosen, reward)
        
        # Store history for analysis
        self.uncertainty_history.append(self.get_all_uncertainties())
        self.relational_uncertainty_history.append(self.get_all_relational_uncertainties())
        self.position_history.append(self.get_all_positions())
        self.Q_history.append(self.Q.copy())
        self.unexpected_uncertainty_history.append(self.unexpected_uncertainty)
    
    def update_uncertainties(self, chosen, unchosen, reward):
        """
        Update uncertainty estimates for stimuli
        
        Parameters:
            - chosen (int): index of chosen stimulus
            - unchosen (int): index of unchosen stimulus
            - reward (int): 1 for reward, 0 for no reward
        """
        # Calculate prediction error
        if reward == 1:
            # Prediction error for correct response
            if self.Q[chosen] < self.Q[unchosen]:
                prediction_error = 1.0  # Large error if model predicted incorrect choice
            else:
                prediction_error = abs(1 - (self.Q[chosen] - self.Q[unchosen]))
        else:
            # Prediction error for incorrect response
            if self.Q[chosen] > self.Q[unchosen]:
                prediction_error = 1.0  # Large error if model predicted correct choice
            else:
                prediction_error = abs(1 - (self.Q[unchosen] - self.Q[chosen]))
        
        # Update uncertainty based on prediction error
        # Higher prediction error = higher uncertainty
        uncertainty_learning_rate = 0.2
        self.uncertainty[chosen] = (1 - uncertainty_learning_rate) * self.uncertainty[chosen] + uncertainty_learning_rate * prediction_error
        self.uncertainty[unchosen] = (1 - uncertainty_learning_rate) * self.uncertainty[unchosen] + uncertainty_learning_rate * prediction_error
        
        # Decay uncertainty for other stimuli
        for j in range(self.n_stimuli):
            if j != chosen and j != unchosen:
                self.uncertainty[j] *= 0.99


class QLearnTI_Lite:
    """
    Lightweight version of QLearnTI model for optimization
    """
    
    def __init__(self, n_stimuli, alpha=0.1, gamma=0.95, tau=0.05, xi=0.95):
        """Initialize lightweight Q-learning model"""
        self.n_stimuli = n_stimuli
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.xi = xi
        
        # Initialize Q-values
        self.Q = np.zeros(n_stimuli)
        
        # Initialize reward tracking
        self.R = np.ones(n_stimuli)
        self.N = np.ones(n_stimuli)
        
        # Initialize uncertainty estimates
        self.uncertainty = np.ones(n_stimuli)
    
    def choose(self, available_stimuli):
        """Choice policy using softmax with exploration"""
        if np.random.random() < self.tau:
            return np.random.choice(available_stimuli)
        
        q_values = np.array([self.Q[i] for i in available_stimuli])
        temperature = 0.5
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / temperature)
        probabilities = exp_q / np.sum(exp_q)
        
        if len(available_stimuli) > 1:
            chosen_idx = np.random.choice(len(available_stimuli), p=probabilities)
            return available_stimuli[chosen_idx]
        else:
            return available_stimuli[0]
    
    def update(self, chosen, unchosen, reward, probability, threshold=0.75):
        """Simplified Q-learning update"""
        # Memory decay step
        self.Q = self.Q * self.xi
        self.R = self.R * self.xi
        self.N = self.N * self.xi
        
        # Update reward tracking
        if reward == 1:
            self.R[chosen] = self.R[chosen] + 1
            self.R[unchosen] = self.R[unchosen] + 1
        else:
            self.N[chosen] = self.N[chosen] + 1
            self.N[unchosen] = self.N[unchosen] + 1
        
        # Calculate target difference
        target_diff = 1.0 if reward == 1 else -1.0
        
        # Adjust learning rate based on confidence
        alpha_mod = 2.0 if probability < threshold else 1.0
        
        # Update Q-values
        alpha_chosen = self.alpha * alpha_mod
        alpha_unchosen = self.alpha * alpha_mod
        
        self.Q[chosen] += alpha_chosen * (target_diff - (self.Q[chosen] - self.Q[unchosen]))
        self.Q[unchosen] -= alpha_unchosen * (target_diff - (self.Q[chosen] - self.Q[unchosen]))
        
        # Simplified implicit inference
        for j in range(self.n_stimuli):
            if j != chosen and j != unchosen:
                if min(self.Q[chosen], self.Q[unchosen]) <= self.Q[j] <= max(self.Q[chosen], self.Q[unchosen]):
                    # Implicit update with smaller learning rate
                    alpha_implicit = self.alpha * 0.2
                    if self.Q[chosen] > self.Q[unchosen]:
                        target = self.Q[j] + alpha_implicit
                    else:
                        target = self.Q[j] - alpha_implicit
                    
                    self.Q[j] = target


def compare_q_model_to_one_rat(all_data_df, rat, n_simulations=100, alpha=0.1, 
                               gamma=0.95, tau=0.05, xi=0.95, threshold=0.75):
    """
    Compare Q-learning model predictions to actual rat choices
    
    Parameters:
        - all_data_df: DataFrame with choice data
        - rat: rat identifier
        - n_simulations: number of simulations per trial
        - alpha, gamma, tau, xi, threshold: model parameters
        
    Returns:
        - final_model: Final model after all days
        - all_models: Dictionary of models for each day
        - match_rates: List of match rates for each day
    """
    all_models = {}
    
    # Track global stimulus states
    global_Q = {}
    global_R = {}
    global_N = {}
    global_uncertainty = {}
    
    # Store match rates for each day
    match_rates = []
    
    # Process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        print(f"\n  Rat {rat} Day {day}", end="", flush=True)
        
        # Extract relevant data
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        
        # Identify which stimuli are present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1  # +1 because of 0-indexing
        
        # Initialize a new model for this day
        model = QLearnTI(n_stimuli, rat, day, alpha=alpha, gamma=gamma, tau=tau, xi=xi)
        
        # Transfer state from previous days
        for stim_idx in range(n_stimuli):
            if stim_idx in global_Q:
                model.Q[stim_idx] = global_Q[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
                model.uncertainty[stim_idx] = global_uncertainty[stim_idx]
        
        # Reset histories
        model.uncertainty_history = [model.get_all_uncertainties()]
        model.relational_uncertainty_history = [model.get_all_relational_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.Q_history = [model.Q.copy()]
        
        # Process the trials for today
        participant_choices = np.column_stack((chosen_idx, unchosen_idx))
        n_trials = len(participant_choices)
        matches = np.zeros(n_trials)
        
        for t in range(n_trials):
            if t % 100 == 0 or (n_trials < 100 and t % 10 == 0):
                print(".", end="", flush=True)
            
            chosen_idx, other_idx = participant_choices[t]
            
            # Validate indices
            if not (0 <= chosen_idx < n_stimuli) or not (0 <= other_idx < n_stimuli):
                print(f"Day {day}, Trial {t}: Invalid indices - chosen {chosen_idx} unchosen {other_idx}")
                continue

            # Run multiple simulations to get choice probability
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx, other_idx])
                model_choices[sim] = model_choice
            
            # Calculate match rate with real choices
            model_match_rate = np.mean(model_choices == chosen_idx)
            matches[t] = model_match_rate

            # Update model based on actual feedback
            reward = 1 if chosen_idx < other_idx else 0  # Like Betasort: lower index = correct
            model.update(chosen_idx, other_idx, reward, model_match_rate, threshold=threshold)
        
        # Calculate cumulative match rate
        cumulative_match_rate = np.mean(matches)
        match_rates.append(cumulative_match_rate)
        
        # Store all the stuff for this day
        all_models[day] = model

        # Update global states
        for stim_idx in range(n_stimuli):
            global_Q[stim_idx] = model.Q[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
            global_uncertainty[stim_idx] = model.uncertainty[stim_idx]
    
    # Return the models
    final_day = max(all_models.keys())
    return all_models[final_day], all_models, match_rates


def find_optimal_q_parameters(all_data_df, rat, n_simulations=50, verbose=True, max_iter=50):
    """
    Find optimal parameters for Q-learning model using differential evolution
    
    Parameters:
        - all_data_df: DataFrame with choice data
        - rat: rat identifier
        - n_simulations: number of simulations per choice
        - verbose: whether to print progress
        - max_iter: maximum iterations for optimization
        
    Returns:
        - best parameters and performance
    """
    import time
    
    def evaluate_model(params, all_data, rat, n_sims=n_simulations):
        """Objective function for optimization"""
        alpha, gamma, tau, xi, threshold = params
        
        if verbose:
            start_time = time.time()
        
        global_Q = {}
        global_R = {}
        global_N = {}
        global_uncertainty = {}
        day_match_rates = []
        
        # Process each day
        for day, day_data in all_data_df.groupby('Day'):
            chosen_idx = day_data["first"].values
            unchosen_idx = day_data["second"].values
            present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
            n_stimuli = max(present_stimuli) + 1
            
            # Initialize model
            model = QLearnTI_Lite(n_stimuli, alpha=alpha, gamma=gamma, tau=tau, xi=xi)
            
            # Transfer state from previous days
            for stim_idx in range(n_stimuli):
                if stim_idx in global_Q:
                    model.Q[stim_idx] = global_Q[stim_idx]
                    model.R[stim_idx] = global_R[stim_idx]
                    model.N[stim_idx] = global_N[stim_idx]
                    model.uncertainty[stim_idx] = global_uncertainty[stim_idx]
            
            # Process trials
            participant_choices = np.column_stack((chosen_idx, unchosen_idx))
            n_trials = len(participant_choices)
            matches = np.zeros(n_trials)
            
            for t in range(n_trials):
                chosen_idx, other_idx = participant_choices[t]
                
                if not (0 <= chosen_idx < n_stimuli) or not (0 <= other_idx < n_stimuli):
                    continue
                
                # Run simulations
                model_choices = np.zeros(n_sims)
                for sim in range(n_sims):
                    model_choice = model.choose([chosen_idx, other_idx])
                    model_choices[sim] = model_choice
                
                model_match_rate = np.mean(model_choices == chosen_idx)
                matches[t] = model_match_rate
                
                # Update model
                reward = 1 if chosen_idx < other_idx else 0
                model.update(chosen_idx, other_idx, reward, model_match_rate, threshold=threshold)
            
            # Store match rate and update global state
            day_match_rates.append(np.mean(matches))
            
            for stim_idx in range(n_stimuli):
                global_Q[stim_idx] = model.Q[stim_idx]
                global_R[stim_idx] = model.R[stim_idx]
                global_N[stim_idx] = model.N[stim_idx]
        
        # Calculate average match rate
        avg_match_rate = np.mean(day_match_rates)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"α={alpha:.3f}, γ={gamma:.3f}, τ={tau:.3f}, ξ={xi:.3f}, θ={threshold:.3f} → {avg_match_rate:.4f} in {elapsed:.2f}s")
        
        # Return negative match rate for minimization
        return -avg_match_rate
    
    if verbose:
        print("\n" + "="*50)
        print(f"Starting optimization for Q-learning parameters")
        print("="*50 + "\n")
    
    # Parameter bounds:
    # alpha (learning rate): 0.01-0.5
    # gamma (discount factor): 0.7-0.99
    # tau (exploration): 0.01-0.2
    # xi (memory decay): 0.8-0.99
    # threshold: 0.6-0.9
    bounds = [(0.01, 0.5), (0.7, 0.99), (0.01, 0.2), (0.8, 0.99), (0.6, 0.9)]
    
    result = differential_evolution(
        evaluate_model,
        bounds=bounds,
        args=(all_data_df, rat),
        popsize=15,
        maxiter=max_iter,
        mutation=(0.5, 1.5),
        recombination=0.7,
        disp=verbose
    )
    
    best_alpha, best_gamma, best_tau, best_xi, best_threshold = result.x
    best_performance = -result.fun
    
    if verbose:
        print("\n" + "="*50)
        print("Optimization results:")
        print(f"Best parameters: α={best_alpha:.4f}, γ={best_gamma:.4f}, τ={best_tau:.4f}, ξ={best_xi:.4f}, θ={best_threshold:.4f}")
        print(f"Best performance: {best_performance:.4f}")
        print("="*50)
    
    return best_alpha, best_gamma, best_tau, best_xi, best_threshold, best_performance


def check_transitive_inference_q(model, n_simulations=100):
    """
    Check transitive inference performance using Q-learning model
    
    Parameters:
        - model: Trained Q-learning model
        - n_simulations: Number of simulations per choice
        
    Returns:
        - Dictionary with performance on all possible pairs
    """
    results = {}
    
    # Check all possible stimulus pairs
    for chosen_idx in range(0, 4):  # max out at 3
        for other_idx in range(chosen_idx + 1, 5):
            # Skip same element
            if chosen_idx == other_idx:
                continue
            
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx, other_idx])
                model_choices[sim] = model_choice
            
            # Calculate how often model chooses the "correct" stimulus (lower index)
            model_correct_rate = np.mean(model_choices == chosen_idx)
            
            results[(chosen_idx, other_idx)] = model_correct_rate
            
    return results


def plot_q_learning_positions(model, stimulus_labels=None, save=None):
    """
    Plot positions (Q-values scaled to 0-1) over trials
    
    Parameters:
        - model: Q-learning model with position history
        - stimulus_labels: labels for stimuli
        - save: path to save figure
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    plt.figure(figsize=(12, 8))
    position_array = np.array(model.position_history)
    
    for i in range(model.n_stimuli):
        plt.plot(position_array[:, i], label=stimulus_labels[i])
        
    plt.xlabel("Trial")
    plt.ylabel("Estimated Position")
    plt.title("Estimated Stimulus Positions from Q-Learning Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def plot_uncertainty_q(model, uncertainty_type='stimulus', stimulus_labels=None, save=None):
    """
    Plot uncertainty over trials
    
    Parameters:
        - model: Q-learning model with uncertainty history
        - uncertainty_type: 'stimulus' or 'relational'
        - stimulus_labels: labels for stimuli
        - save: path to save figure
    """
    if uncertainty_type == 'stimulus':
        # Plot stimulus uncertainty
        uncertainty_array = np.array(model.uncertainty_history)
        if stimulus_labels is None:
            stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
        
        plt.figure(figsize=(12, 8))
        
        for i in range(model.n_stimuli):
            plt.plot(uncertainty_array[:, i], label=stimulus_labels[i])
        
        plt.xlabel("Trial")
        plt.ylabel("Uncertainty")
        plt.title("Trial-by-Trial Stimulus Uncertainty from Q-Learning Model")
        
    else:
        # Plot relational uncertainty
        uncertainty_array = np.array(model.relational_uncertainty_history)
        if stimulus_labels is None:
            stimulus_labels = [f"Stimuli {i}-{i+1}" for i in range(model.n_stimuli-1)]
        
        plt.figure(figsize=(12, 8))
        
        for i in range(model.n_stimuli-1):
            plt.plot(uncertainty_array[:, i], label=stimulus_labels[i])
        
        plt.xlabel("Trial")
        plt.ylabel("Uncertainty")
        plt.title("Trial-by-Trial Relational Uncertainty from Q-Learning Model")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def plot_q_values(model, stimulus_labels=None, save=None):
    """
    Plot Q-values over trials
    
    Parameters:
        - model: Q-learning model with Q-value history
        - stimulus_labels: labels for stimuli
        - save: path to save figure
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    plt.figure(figsize=(12, 8))
    q_array = np.array(model.Q_history)
    
    for i in range(model.n_stimuli):
        plt.plot(q_array[:, i], label=stimulus_labels[i])
        
    plt.xlabel("Trial")
    plt.ylabel("Q-Value")
    plt.title("Q-Values from Q-Learning Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()