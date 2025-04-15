import numpy as np

class RWGeneralization:
    def __init__(self, n_stimuli, alpha=0.1, beta=0.1, generalization_factor=0.5):
        self.n_stimuli = n_stimuli
        self.alpha = alpha  # Learning rate
        self.beta = beta  # Generalization rate
        self.generalization_factor = generalization_factor
        self.values = np.ones(n_stimuli) * 0.5  # Initialize values at 0.5
        
        # History tracking
        self.value_history = [self.values.copy()]
        
    def choose(self, available_stimuli):
        """Softmax choice policy"""
        values = np.array([self.values[s] for s in available_stimuli])
        probs = np.exp(values) / np.sum(np.exp(values))
        return np.random.choice(available_stimuli, p=probs)
    
    def update(self, chosen, unchosen, reward):
        # Update chosen stimulus value
        prediction_error = reward - self.values[chosen]
        self.values[chosen] += self.alpha * prediction_error
        
        # Update unchosen stimulus with inverse prediction error
        inverse_prediction_error = (1-reward) - self.values[unchosen]
        self.values[unchosen] += self.alpha * inverse_prediction_error
        
        # Generalization to other stimuli
        for i in range(self.n_stimuli):
            if i != chosen and i != unchosen:
                # Generalize based on similarity to chosen/unchosen
                chosen_similarity = 1/(abs(i-chosen)+1)
                unchosen_similarity = 1/(abs(i-unchosen)+1)
                
                # Update based on generalization
                self.values[i] += self.beta * chosen_similarity * prediction_error
                self.values[i] += self.beta * unchosen_similarity * inverse_prediction_error
        
        # Store value history
        self.value_history.append(self.values.copy())
    
    def get_uncertainty(self, stim1, stim2):
        """Uncertainty as difference between stimulus values"""
        value_diff = abs(self.values[stim1] - self.values[stim2])
        # Convert to uncertainty (closer values = higher uncertainty)
        return 1 - value_diff