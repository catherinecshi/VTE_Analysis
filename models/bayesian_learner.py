import numpy as np

class BayesianLearner:
    def __init__(self, n_stimuli, prior_mean=0.5, prior_var=1.0):
        self.n_stimuli = n_stimuli
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        
        # For each stimulus, track mean and variance of value
        self.means = np.ones(n_stimuli) * prior_mean
        self.variances = np.ones(n_stimuli) * prior_var
        
        # History tracking
        self.mean_history = [self.means.copy()]
        self.variance_history = [self.variances.copy()]
    
    def choose(self, available_stimuli, n_samples=100):
        """Thompson sampling choice with improved numerical stability"""
        # Ensure capacity for all stimuli
        self._ensure_capacity(max(available_stimuli) + 1)
        
        sampled_values = np.zeros(len(available_stimuli))
        
        for i, stim in enumerate(available_stimuli):
            # Ensure variance is positive to prevent NaN from sqrt
            variance = max(1e-6, float(self.variances[stim]))
            
            # Sample from distribution for each stimulus
            sampled_values[i] = np.random.normal(self.means[stim], np.sqrt(variance))
        
        # Check for NaN values
        if np.any(np.isnan(sampled_values)):
            print(f"Warning: NaN detected in sampled values: {sampled_values}")
            sampled_values = np.nan_to_num(sampled_values, nan=self.prior_mean)
            
        return available_stimuli[np.argmax(sampled_values)]
    
    def update(self, chosen, unchosen, reward):
        # Ensure capacity for all stimuli
        self._ensure_capacity(max(chosen, unchosen) + 1)
        
        # Bayesian update for chosen stimulus
        k = self.variances[chosen] / (self.variances[chosen] + 0.1)  # Kalman gain
        self.means[chosen] = self.means[chosen] + k * (reward - self.means[chosen])
        self.variances[chosen] = (1 - k) * self.variances[chosen]
        
        # Bayesian update for unchosen stimulus (with inverse reward)
        k = self.variances[unchosen] / (self.variances[unchosen] + 0.1)
        self.means[unchosen] = self.means[unchosen] + k * ((1-reward) - self.means[unchosen])
        self.variances[unchosen] = (1 - k) * self.variances[unchosen]
        
        # Ensure variances remain positive
        self.variances = np.maximum(self.variances, 1e-6)
        
        # Check for NaN values
        if np.any(np.isnan(self.means)) or np.any(np.isnan(self.variances)):
            print(f"Warning: NaN values detected after update")
            self.means = np.nan_to_num(self.means, nan=self.prior_mean)
            self.variances = np.nan_to_num(self.variances, nan=self.prior_var)
            self.variances = np.maximum(self.variances, 1e-6)  # Ensure positive after NaN replacement
            
        # History tracking
        self.mean_history.append(self.means.copy())
        self.variance_history.append(self.variances.copy())
    
    def get_uncertainty(self, stim1, stim2, n_samples=1000):
        """Calculate uncertainty between stimulus pair"""
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(stim1, stim2) + 1)
        
        # Ensure variances are positive to prevent NaN from sqrt
        var1 = max(1e-6, float(self.variances[stim1]))
        var2 = max(1e-6, float(self.variances[stim2]))
        
        samples1 = np.random.normal(self.means[stim1], np.sqrt(var1), n_samples)
        samples2 = np.random.normal(self.means[stim2], np.sqrt(var2), n_samples)
        
        # Probability that one is greater than the other
        prob_greater = np.mean(samples1 > samples2)
        
        # Convert to uncertainty (0.5 = maximum uncertainty)
        return 1 - 2 * abs(prob_greater - 0.5)
    
    def _ensure_capacity(self, required_size):
        """Ensure that the model can handle at least required_size stimuli"""
        if required_size > self.n_stimuli:
            # Expand means
            new_means = np.ones(required_size) * self.prior_mean
            new_means[:self.n_stimuli] = self.means
            self.means = new_means
            
            # Expand variances
            new_variances = np.ones(required_size) * self.prior_var
            new_variances[:self.n_stimuli] = self.variances
            self.variances = new_variances
            
            # Update n_stimuli
            self.n_stimuli = required_size
    
    def copy(self):
        """Create a deep copy of this model"""
        copied = BayesianLearner(self.n_stimuli, self.prior_mean, self.prior_var)
        copied.means = self.means.copy()
        copied.variances = self.variances.copy()
        copied.mean_history = [m.copy() for m in self.mean_history]
        copied.variance_history = [v.copy() for v in self.variance_history]
        return copied