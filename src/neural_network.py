import numpy as np

class NeuralNetworkModel:
    def __init__(self, n_stimuli, learning_rate=0.01, hidden_size=10, dropout_rate=0.2):
        self.n_stimuli = n_stimuli
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        # Simple model structure (using numpy for compatibility)
        self.W1 = np.random.randn(n_stimuli, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)
        
        # History
        self.prediction_history = []
    
    def _forward(self, x, apply_dropout=False):
        """Forward pass through the network"""
        # One-hot encode the stimulus
        x_one_hot = np.zeros(self.n_stimuli)
        x_one_hot[x] = 1
        
        # First layer
        h = np.dot(x_one_hot, self.W1) + self.b1
        h = 1 / (1 + np.exp(-h))  # Sigmoid
        
        # Apply dropout if requested
        if apply_dropout:
            mask = (np.random.rand(*h.shape) > self.dropout_rate)
            h = h * mask / (1 - self.dropout_rate)  # Scale to maintain expected value
        
        # Output layer
        y = np.dot(h, self.W2) + self.b2
        return 1 / (1 + np.exp(-y))  # Sigmoid output
    
    def choose(self, available_stimuli):
        """Choose based on predicted values"""
        values = np.array([self._forward(s)[0] for s in available_stimuli])
        return available_stimuli[np.argmax(values)]
    
    def update(self, chosen, unchosen, reward):
        """Update network weights with stochastic gradient descent"""
        # One-hot encode chosen stimulus
        x_one_hot = np.zeros(self.n_stimuli)
        x_one_hot[chosen] = 1
        
        # Forward pass
        h = np.dot(x_one_hot, self.W1) + self.b1
        h = 1 / (1 + np.exp(-h))  # Sigmoid
        y = np.dot(h, self.W2) + self.b2
        y = 1 / (1 + np.exp(-y))  # Sigmoid output
        
        # Backward pass
        d_y = (y - reward)
        d_W2 = np.outer(h, d_y)
        d_b2 = d_y
        d_h = np.dot(d_y, self.W2.T)
        d_h = d_h * h * (1 - h)  # Sigmoid derivative
        d_W1 = np.outer(x_one_hot, d_h)
        d_b1 = d_h
        
        # Update weights
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        
        # Store prediction
        self.prediction_history.append(y[0])
    
    def get_uncertainty(self, stim1, stim2, n_samples=20):
        """Use Monte Carlo dropout for uncertainty estimation"""
        # Run multiple forward passes with dropout
        pred1_samples = [self._forward(stim1, apply_dropout=True)[0] for _ in range(n_samples)]
        pred2_samples = [self._forward(stim2, apply_dropout=True)[0] for _ in range(n_samples)]
        
        # Calculate mean predictions
        mean1 = np.mean(pred1_samples)
        mean2 = np.mean(pred2_samples)
        
        # Calculate variances
        var1 = np.var(pred1_samples)
        var2 = np.var(pred2_samples)
        
        # Calculate uncertainty as a combination of:
        # 1. Overlap between distributions
        # 2. Total variance 
        relative_diff = abs(mean1 - mean2) / np.sqrt(var1 + var2 + 1e-6)
        total_var = (var1 + var2) / 2
        
        # Convert to uncertainty (0-1 scale)
        distribution_uncertainty = np.exp(-relative_diff)
        
        return distribution_uncertainty * (1 + total_var)