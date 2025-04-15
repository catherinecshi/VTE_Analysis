import numpy as np

class BayesianLearner:
    def __init__(self, n_stimuli, prior_mean=0.5, prior_var=1.0):
        self.n_stimuli = n_stimuli
        
        # For each stimulus, track mean and variance of value
        self.means = np.ones(n_stimuli) * prior_mean
        self.variances = np.ones(n_stimuli) * prior_var
        
        # History tracking
        self.mean_history = [self.means.copy()]
        self.variance_history = [self.variances.copy()]
    
    def choose(self, available_stimuli, n_samples=100):
        """Thompson sampling choice"""
        sampled_values = np.zeros(len(available_stimuli))
        
        for i, stim in enumerate(available_stimuli):
            # Sample from distribution for each stimulus
            sampled_values[i] = np.random.normal(self.means[stim], np.sqrt(self.variances[stim]))
        
        return available_stimuli[np.argmax(sampled_values)]
    
    def update(self, chosen, unchosen, reward):
        # Bayesian update for chosen stimulus
        k = self.variances[chosen] / (self.variances[chosen] + 0.1)  # Kalman gain
        self.means[chosen] = self.means[chosen] + k * (reward - self.means[chosen])
        self.variances[chosen] = (1 - k) * self.variances[chosen]
        
        # Bayesian update for unchosen stimulus (with inverse reward)
        k = self.variances[unchosen] / (self.variances[unchosen] + 0.1)
        self.means[unchosen] = self.means[unchosen] + k * ((1-reward) - self.means[unchosen])
        self.variances[unchosen] = (1 - k) * self.variances[unchosen]
        
        # History tracking
        self.mean_history.append(self.means.copy())
        self.variance_history.append(self.variances.copy())
    
    def get_uncertainty(self, stim1, stim2, n_samples=1000):
        """Calculate uncertainty between stimulus pair"""
        samples1 = np.random.normal(self.means[stim1], np.sqrt(self.variances[stim1]), n_samples)
        samples2 = np.random.normal(self.means[stim2], np.sqrt(self.variances[stim2]), n_samples)
        
        # Probability that one is greater than the other
        prob_greater = np.mean(samples1 > samples2)
        
        # Convert to uncertainty (0.5 = maximum uncertainty)
        return 1 - 2 * abs(prob_greater - 0.5)