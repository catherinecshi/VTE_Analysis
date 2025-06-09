import numpy as np

class RWGeneralization:
    def __init__(self, n_stimuli, alpha=0.1, beta=0.1):
        self.n_stimuli = n_stimuli
        self.alpha = alpha  # Learning rate
        self.beta = beta  # Generalization rate
        self.values = np.ones(n_stimuli) * 0.5  # Initialize values at 0.5
        
        # History tracking
        self.value_history = [self.values.copy()]
        
    def choose(self, available_stimuli):
        """Softmax choice policy with numerical stability fixes"""
        self._ensure_capacity(max(available_stimuli) + 1)
        
        # Extract values
        values = np.array([self.values[s] for s in available_stimuli])
        
        # Apply softmax with numerical stability correction
        values_shifted = values - np.max(values)
        exp_values = np.exp(values_shifted)
        sum_exp = np.sum(exp_values)
        
        # Handle any zero-sum case (rare but possible with extreme underflow)
        if sum_exp == 0:
            # Equal probability for all options
            probs = np.ones_like(values) / len(values)
        else:
            probs = exp_values / sum_exp
        
        # Ensure probabilities sum to 1 and no zeros if possible
        probs = np.clip(probs, 1e-10, 1.0)  # Prevent exact zeros
        probs = probs / np.sum(probs)  # Renormalize to guarantee sum=1
        
        if np.isnan(probs).any() or (probs == 0).any():
            return np.random.choice(available_stimuli)
        else:
            # Ensure probabilities sum to 1
            probs = probs / np.sum(probs)  # Renormalize to guarantee sum=1
            return np.random.choice(available_stimuli, p=probs)
    
    def update(self, chosen, unchosen, reward):
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(chosen, unchosen) + 1)
        
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
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(stim1, stim2) + 1)
        
        value_diff = abs(self.values[stim1] - self.values[stim2])
        # Convert to uncertainty (closer values = higher uncertainty)
        return 1 - value_diff
    
    def _ensure_capacity(self, required_size):
        """Ensure that the model can handle at least required_size stimuli"""
        if required_size > self.n_stimuli:
            # Expand values array
            new_values = np.ones(required_size) * 0.5
            new_values[:self.n_stimuli] = self.values
            self.values = new_values
            
            # Update n_stimuli
            self.n_stimuli = required_size
    
    def copy(self):
        """Create a deep copy of this model"""
        copied = RWGeneralization(self.n_stimuli, self.alpha, self.beta)
        copied.values = self.values.copy()
        copied.value_history = [v.copy() for v in self.value_history]
        return copied