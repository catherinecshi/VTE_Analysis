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
        """Softmax choice based on transferred values with improved numerical stability"""
        # Calculate values for each stimulus
        values = np.zeros(len(available_stimuli))
        
        for i, stim in enumerate(available_stimuli):
            # Ensure we're not accessing beyond array bounds
            if stim >= self.n_stimuli:
                # Expand the matrix if needed
                self._ensure_capacity(stim + 1)
                
            # Sum all associations for this stimulus
            for j in range(self.n_stimuli):
                if j != stim:
                    values[i] += self.V[stim, j]
        
        # Check for NaN in values
        if np.any(np.isnan(values)):
            print(f"Warning: NaN detected in values: {values}")
            # Replace NaNs with zeros
            values = np.nan_to_num(values, nan=0.0)
        
        # Apply numerically stable softmax
        # Subtract max for numerical stability
        values = values / self.temp
        values_shifted = values - np.max(values)
        exp_values = np.exp(values_shifted)
        sum_exp = np.sum(exp_values)
        
        # Handle the case where sum is zero (all values are -inf)
        if sum_exp == 0:
            print("Warning: All exponential values are zero, using uniform distribution")
            print("value transfer")
            probs = np.ones(len(available_stimuli)) / len(available_stimuli)
        else:
            probs = exp_values / sum_exp
            
        # Final check to ensure valid probabilities
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            print(f"Warning: Invalid probabilities detected: {probs}")
            print(f"Values after temperature: {values}")
            print(f"Exponential values: {exp_values}")
            print(f"Sum of exponentials: {sum_exp}")
            # Fall back to uniform distribution
            probs = np.ones(len(available_stimuli)) / len(available_stimuli)
        
        # Make sure probabilities sum to 1
        probs = probs / np.sum(probs)
        
        return np.random.choice(available_stimuli, p=probs)
    
    def update(self, chosen, unchosen, reward):
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(chosen, unchosen) + 1)
        
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
        
        # Check for and handle NaN values
        if np.any(np.isnan(self.V)):
            print("Warning: NaN values detected in V matrix after update")
            indices = np.where(np.isnan(self.V))
            for i, j in zip(indices[0], indices[1]):
                print(f"NaN at position ({i}, {j})")
                # Replace NaNs with zeros
                self.V[i, j] = 0.0
        
        # Store history
        self.V_history.append(self.V.copy())
    
    def get_uncertainty(self, stim1, stim2):
        """Calculate uncertainty between a pair of stimuli"""
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(stim1, stim2) + 1)
        
        # Total value for each stimulus
        val1 = np.sum(self.V[stim1, :])
        val2 = np.sum(self.V[stim2, :])
        
        # Handle NaN values
        if np.isnan(val1) or np.isnan(val2):
            print(f"Warning: NaN values in uncertainty calculation: val1={val1}, val2={val2}")
            val1 = 0.0 if np.isnan(val1) else val1
            val2 = 0.0 if np.isnan(val2) else val2
        
        # Difference in values scaled to [0,1]
        # Add small epsilon to avoid division by zero
        diff = abs(val1 - val2) / (val1 + val2 + 1e-10)
        
        # Convert to uncertainty (smaller difference = higher uncertainty)
        return 1 - diff
    
    def _ensure_capacity(self, required_size):
        """Ensure that the model can handle at least required_size stimuli"""
        if required_size > self.n_stimuli:
            # Expand V matrix
            new_V = np.zeros((required_size, required_size))
            new_V[:self.n_stimuli, :self.n_stimuli] = self.V
            self.V = new_V
            
            # Update n_stimuli
            self.n_stimuli = required_size
    
    def copy(self):
        """Create a deep copy of this model"""
        copied = ValueTransferModel(self.n_stimuli, self.alpha, self.transfer_rate, self.temp)
        copied.V = self.V.copy()
        copied.V_history = [v.copy() for v in self.V_history]
        return copied