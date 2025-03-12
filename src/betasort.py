import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
        xi_values = np.arange(0.75, 0.99, 0.01)
    
    if tau_values is None:
        tau_values = np.arange(0.01, 0.25, 0.01)
    
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

def plot_uncertainty(model, stimulus_labels=None):
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
    plt.xticks(range(len(tau_list)), [f"{t:.2f}" for t in tau_list])
    plt.yticks(range(len(xi_list)), [f"{x:.2f}" for x in xi_list])
    
    # text annotations with values
    for i, xi in enumerate(xi_list):
        for j, tau in enumerate(tau_list):
            plt.text(j, i, f"{performance_matrix[i, j]:.3f}",
                     ha="center", va="center", color="w" if performance_matrix[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    plt.show()

def plot_best_parameters(best_model):
    plot_positions(best_model)
    plot_uncertainty(best_model)
    plot_beta_distributions(best_model)