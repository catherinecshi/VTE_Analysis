import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class Betasort:
    """
    Implementation of betasort model according to pseudocode from "Implicit Value Updating Explains TI Performance"
    Greg Jensen et al
    """
    
    def __init__(self, n_stimuli, tau=0.05, xi=0.95):
        """
        initialise betasort model
        
        Parameters:
            - n_stimuli (int): number of stimuli in the list
            - tau (float): noise parameter
            - xi (float): recall parameter
        """
        
        self.n_stimuli = n_stimuli
        self.tau = tau
        self.xi = xi

        # initialise memory arrays
        self.U = np.ones(n_stimuli) # upper parameter
        self.L = np.ones(n_stimuli) # lower parameter
        self.R = np.ones(n_stimuli) # rewards
        self.N = np.ones(n_stimuli) # non rewards
    
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
            print(a, b, "a + b < 2")
            return 1.0 # maximum uncertainty
        
        return (a * b) / ((a + b)**2 * (a + b + 1))

    def get_all_stimulus_uncertainties(self):
        """Get uncertainty values for all stimuli"""
        return np.array([self.get_uncertainty_stimulus(i) for i in range(self.n_stimuli)])

    def update(self, chosen, unchosen, reward):
        """
        implements first updating policy
        
        Parameters:
            - chosen (int): index of chosen stimulus
            - unchosen (int): index of unchosen stimulus
            - reward (int): 1 for reward and 0 otherwise
        """
        
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
            # consolidate all stimuli
            self.U = self.U + V
            self.L = self.L + (1 - V)
        else:
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
    
    
def analyze_data