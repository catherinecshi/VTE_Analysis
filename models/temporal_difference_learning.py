import numpy as np

class TDLambdaModel:
    def __init__(self, n_stimuli, alpha=0.1, gamma=0.9, lambda_=0.6, temp=1.0):
        self.n_stimuli = n_stimuli
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.lambda_ = lambda_  # Eligibility trace decay
        self.temp = temp  # Temperature for softmax
        
        # Values and eligibility traces
        self.values = np.ones(n_stimuli) * 0.5
        self.e_traces = np.zeros(n_stimuli)
        
        # History tracking
        self.value_history = [self.values.copy()]
        self.e_trace_history = [self.e_traces.copy()]
        
        # Tracking for uncertainty
        self.value_variance = np.ones(n_stimuli) * 0.25
    
    def choose(self, available_stimuli):
        """Softmax choice policy with improved numerical stability"""
        # Ensure capacity for all stimuli
        self._ensure_capacity(max(available_stimuli) + 1)
        
        values = np.array([self.values[s] for s in available_stimuli])
        
        # Check for NaN values
        if np.any(np.isnan(values)):
            print(f"Warning: NaN detected in values: {values}")
            values = np.nan_to_num(values, nan=0.5)
            
        # Apply softmax with numerical stability
        values = values / self.temp
        values_shifted = values - np.max(values)
        exp_values = np.exp(values_shifted)
        sum_exp = np.sum(exp_values)
        
        # Handle edge cases
        if sum_exp == 0 or np.isnan(sum_exp):
            print("Warning: All exponential values are zero or NaN, using uniform distribution")
            print("td lambda")
            probs = np.ones(len(available_stimuli)) / len(available_stimuli)
        else:
            probs = exp_values / sum_exp
            
        # Final validity check
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            print(f"Warning: Invalid probabilities: {probs}")
            probs = np.ones(len(available_stimuli)) / len(available_stimuli)
            
        # Ensure probabilities sum to 1
        probs = probs / np.sum(probs)
        
        return np.random.choice(available_stimuli, p=probs)
    
    def update(self, chosen, unchosen, reward):
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(chosen, unchosen) + 1)
        
        # Update eligibility traces
        for i in range(self.n_stimuli):
            if i == chosen:
                self.e_traces[i] = 1.0  # Set trace for chosen stimulus
            else:
                self.e_traces[i] *= self.gamma * self.lambda_  # Decay all other traces
        
        # Calculate TD error (prediction error)
        td_error = reward - self.values[chosen]
        
        # Update values using eligibility traces
        for i in range(self.n_stimuli):
            # Update value
            old_value = self.values[i]
            self.values[i] += self.alpha * td_error * self.e_traces[i]
            
            # Update variance estimate for uncertainty
            self.value_variance[i] = 0.9 * self.value_variance[i] + 0.1 * (self.values[i] - old_value)**2
        
        # Check for NaN values
        if np.any(np.isnan(self.values)) or np.any(np.isnan(self.e_traces)) or np.any(np.isnan(self.value_variance)):
            print(f"Warning: NaN values detected after update")
            self.values = np.nan_to_num(self.values, nan=0.5)
            self.e_traces = np.nan_to_num(self.e_traces, nan=0.0)
            self.value_variance = np.nan_to_num(self.value_variance, nan=0.25)
            
        # Store history
        self.value_history.append(self.values.copy())
        self.e_trace_history.append(self.e_traces.copy())
    
    def get_uncertainty(self, stim1, stim2, n_samples=1000):
        """Calculate uncertainty for a stimulus pair using value variances"""
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(stim1, stim2) + 1)
        
        # Ensure valid variance values (prevent negative values)
        var1 = max(0.001, self.value_variance[stim1])
        var2 = max(0.001, self.value_variance[stim2])
        
        # Use variances to generate distributions and calculate overlap
        samples1 = np.random.normal(self.values[stim1], np.sqrt(var1), n_samples)
        samples2 = np.random.normal(self.values[stim2], np.sqrt(var2), n_samples)
        
        # Probability that one value is greater than the other
        prob_greater = np.mean(samples1 > samples2)
        
        # Convert to uncertainty measure (0.5 = maximum uncertainty)
        return 1 - 2 * abs(prob_greater - 0.5)
    
    def _ensure_capacity(self, required_size):
        """Ensure that the model can handle at least required_size stimuli"""
        if required_size > self.n_stimuli:
            # Expand values array
            new_values = np.ones(required_size) * 0.5
            new_values[:self.n_stimuli] = self.values
            self.values = new_values
            
            # Expand e_traces
            new_e_traces = np.zeros(required_size)
            new_e_traces[:self.n_stimuli] = self.e_traces
            self.e_traces = new_e_traces
            
            # Expand value_variance
            new_value_variance = np.ones(required_size) * 0.25
            new_value_variance[:self.n_stimuli] = self.value_variance
            self.value_variance = new_value_variance
            
            # Update n_stimuli
            self.n_stimuli = required_size
    
    def copy(self):
        """Create a deep copy of this model"""
        copied = TDLambdaModel(self.n_stimuli, self.alpha, self.gamma, self.lambda_, self.temp)
        copied.values = self.values.copy()
        copied.e_traces = self.e_traces.copy()
        copied.value_variance = self.value_variance.copy()
        copied.value_history = [v.copy() for v in self.value_history]
        copied.e_trace_history = [e.copy() for e in self.e_trace_history]
        return copied