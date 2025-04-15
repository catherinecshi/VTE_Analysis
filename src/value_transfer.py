import numpy as np

class ValueTransferModel:
    def __init__(self, n_stimuli, alpha=0.1, transfer_rate=0.3, temp=1.0):
        self.n_stimuli = n_stimuli
        self.alpha = alpha  # Learning rate
        self.transfer_rate = transfer_rate  # Value transfer parameter
        self.temp = temp  # Temperature for softmax
        
        # Direct associative strengths
        self.V = np.zeros((n_stimuli, n_stimuli))
        
        # History tracking
        self.V_history = [self.V.copy()]
    
    def choose(self, available_stimuli):
        """Softmax choice based on transferred values"""
        values = np.zeros(len(available_stimuli))
        
        for i, stim in enumerate(available_stimuli):
            for j in range(self.n_stimuli):
                if j != stim:
                    # Total value includes direct + transferred value
                    values[i] += self.V[stim, j]
        
        # Apply softmax
        values = values / self.temp
        probs = np.exp(values) / np.sum(np.exp(values))
        
        return np.random.choice(available_stimuli, p=probs)
    
    def update(self, chosen, unchosen, reward):
        # Update direct association for chosen
        if reward == 1:
            self.V[chosen, unchosen] += self.alpha * (1 - self.V[chosen, unchosen])
            self.V[unchosen, chosen] += self.alpha * (0 - self.V[unchosen, chosen])
        else:
            self.V[chosen, unchosen] += self.alpha * (0 - self.V[chosen, unchosen])
            self.V[unchosen, chosen] += self.alpha * (1 - self.V[unchosen, chosen])
        
        # Value transfer to other stimuli (from both chosen and unchosen)
        for i in range(self.n_stimuli):
            if i != chosen and i != unchosen:
                # Transfer from chosen to i
                self.V[i, chosen] += self.transfer_rate * self.V[i, unchosen] * self.V[unchosen, chosen]
                
                # Transfer from unchosen to i
                self.V[i, unchosen] += self.transfer_rate * self.V[i, chosen] * self.V[chosen, unchosen]
        
        # Store history
        self.V_history.append(self.V.copy())
    
    def get_uncertainty(self, stim1, stim2):
        """Calculate uncertainty between a pair of stimuli"""
        # Total value for each stimulus
        val1 = np.sum(self.V[stim1, :])
        val2 = np.sum(self.V[stim2, :])
        
        # Difference in values scaled to [0,1]
        diff = abs(val1 - val2) / (val1 + val2 + 1e-6)
        
        # Convert to uncertainty (smaller difference = higher uncertainty)
        return 1 - diff