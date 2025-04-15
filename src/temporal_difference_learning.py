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
        """Softmax choice policy"""
        values = np.array([self.values[s] for s in available_stimuli])
        
        # Apply softmax
        values = values / self.temp
        probs = np.exp(values) / np.sum(np.exp(values))
        
        return np.random.choice(available_stimuli, p=probs)
    
    def update(self, chosen, unchosen, reward):
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
        
        # Store history
        self.value_history.append(self.values.copy())
        self.e_trace_history.append(self.e_traces.copy())
    
    def get_uncertainty(self, stim1, stim2, n_samples=1000):
        """Calculate uncertainty for a stimulus pair using value variances"""
        # Use variances to generate distributions and calculate overlap
        samples1 = np.random.normal(self.values[stim1], np.sqrt(self.value_variance[stim1]), n_samples)
        samples2 = np.random.normal(self.values[stim2], np.sqrt(self.value_variance[stim2]), n_samples)
        
        # Probability that one value is greater than the other
        prob_greater = np.mean(samples1 > samples2)
        
        # Convert to uncertainty measure (0.5 = maximum uncertainty)
        return 1 - 2 * abs(prob_greater - 0.5)