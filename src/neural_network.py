import numpy as np

class NeuralNetworkModel:
    def __init__(self, n_stimuli, learning_rate=0.01, hidden_size=10, dropout_rate=0.2):
        self.n_stimuli = n_stimuli
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        
        # Simple model structure (using numpy for compatibility)
        self.W1 = np.random.randn(n_stimuli, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)
        
        # History
        self.prediction_history = []
    
    def _forward(self, x, apply_dropout=False):
        """Forward pass through the network with protection against numerical instability"""
        # Ensure capacity for this stimulus
        self._ensure_capacity(x + 1)
        
        # One-hot encode the stimulus
        x_one_hot = np.zeros(self.n_stimuli)
        x_one_hot[x] = 1
        
        # First layer
        h = np.dot(x_one_hot, self.W1) + self.b1
        
        # Check for NaN in h
        if np.any(np.isnan(h)):
            print(f"Warning: NaN in hidden layer before activation")
            h = np.nan_to_num(h, nan=0.0)
            
        # Sigmoid with clipping to prevent extreme values
        h = 1 / (1 + np.exp(-np.clip(h, -30, 30)))
        
        # Apply dropout if requested
        if apply_dropout:
            mask = (np.random.rand(*h.shape) > self.dropout_rate)
            h = h * mask / (1 - self.dropout_rate)  # Scale to maintain expected value
        
        # Output layer
        y = np.dot(h, self.W2) + self.b2
        
        # Check for NaN in y
        if np.any(np.isnan(y)):
            print(f"Warning: NaN in output layer before activation")
            y = np.nan_to_num(y, nan=0.0)
            
        # Sigmoid with clipping
        y_out = 1 / (1 + np.exp(-np.clip(y, -30, 30)))
        
        # Final NaN check
        if np.any(np.isnan(y_out)):
            print(f"Warning: NaN in final output")
            y_out = np.nan_to_num(y_out, nan=0.5)
            
        return y_out
    
    def choose(self, available_stimuli):
        """Choose based on predicted values with stability checks"""
        # Ensure capacity for all stimuli
        self._ensure_capacity(max(available_stimuli) + 1)
        
        values = np.array([self._forward(s)[0] for s in available_stimuli])
        
        # Check for NaN or Inf values
        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
            print(f"Warning: Invalid values in choose: {values}")
            values = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
            
        # If all values are identical, choose randomly to prevent bias
        if np.all(values == values[0]):
            return np.random.choice(available_stimuli)
            
        return available_stimuli[np.argmax(values)]
    
    def update(self, chosen, unchosen, reward):
        """Update network weights with stochastic gradient descent and numerical stability checks"""
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(chosen, unchosen) + 1)
        
        # One-hot encode chosen stimulus
        x_one_hot = np.zeros(self.n_stimuli)
        x_one_hot[chosen] = 1
        
        # Forward pass with clipping to prevent extreme values
        h = np.dot(x_one_hot, self.W1) + self.b1
        h = np.clip(h, -30, 30)  # Clipping before activation
        h = 1 / (1 + np.exp(-h))  # Sigmoid
        
        y = np.dot(h, self.W2) + self.b2
        y = np.clip(y, -30, 30)  # Clipping before activation
        y = 1 / (1 + np.exp(-y))  # Sigmoid output
        
        # Backward pass with gradient clipping
        d_y = (y - reward)
        d_W2 = np.outer(h, d_y)
        d_b2 = d_y
        d_h = np.dot(d_y, self.W2.T)
        d_h = d_h * h * (1 - h)  # Sigmoid derivative
        d_W1 = np.outer(x_one_hot, d_h)
        d_b1 = d_h
        
        # Clip gradients to prevent extreme updates
        d_W2 = np.clip(d_W2, -1.0, 1.0)
        d_b2 = np.clip(d_b2, -1.0, 1.0)
        d_W1 = np.clip(d_W1, -1.0, 1.0)
        d_b1 = np.clip(d_b1, -1.0, 1.0)
        
        # Update weights
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        
        # Check for NaN values in weights and biases
        if (np.any(np.isnan(self.W1)) or np.any(np.isnan(self.b1)) or 
            np.any(np.isnan(self.W2)) or np.any(np.isnan(self.b2))):
            print("Warning: NaN values detected in network parameters")
            self.W1 = np.nan_to_num(self.W1, nan=0.0)
            self.b1 = np.nan_to_num(self.b1, nan=0.0)
            self.W2 = np.nan_to_num(self.W2, nan=0.0)
            self.b2 = np.nan_to_num(self.b2, nan=0.0)
            
        # Store prediction - handle potential NaN
        prediction = float(y[0])
        if np.isnan(prediction) or np.isinf(prediction):
            prediction = 0.5
        self.prediction_history.append(prediction)
    
    def get_uncertainty(self, stim1, stim2, n_samples=20):
        """Use Monte Carlo dropout for uncertainty estimation with stability checks"""
        # Ensure capacity for these stimuli
        self._ensure_capacity(max(stim1, stim2) + 1)
        
        # Run multiple forward passes with dropout
        pred1_samples = []
        pred2_samples = []
        
        for _ in range(n_samples):
            # Run forward passes with dropout
            pred1 = self._forward(stim1, apply_dropout=True)[0]
            pred2 = self._forward(stim2, apply_dropout=True)[0]
            
            # Ensure values are valid
            if not (np.isnan(pred1) or np.isnan(pred2) or np.isinf(pred1) or np.isinf(pred2)):
                pred1_samples.append(pred1)
                pred2_samples.append(pred2)
        
        # If we have no valid samples, use default values
        if len(pred1_samples) == 0:
            print(f"Warning: No valid samples for uncertainty calculation")
            return 0.5  # Maximum uncertainty
        
        # Convert to numpy arrays
        pred1_samples = np.array(pred1_samples)
        pred2_samples = np.array(pred2_samples)
        
        # Calculate mean predictions with minimum sample check
        mean1 = np.mean(pred1_samples)
        mean2 = np.mean(pred2_samples)
        
        # Calculate variances with minimum sample check
        var1 = np.var(pred1_samples) if len(pred1_samples) > 1 else 0.25
        var2 = np.var(pred2_samples) if len(pred2_samples) > 1 else 0.25
        
        # Ensure variances are positive
        var1 = max(1e-6, var1)
        var2 = max(1e-6, var2)
        
        # Calculate uncertainty as a combination of:
        # 1. Overlap between distributions
        # 2. Total variance
        relative_diff = abs(mean1 - mean2) / np.sqrt(var1 + var2 + 1e-6)
        total_var = (var1 + var2) / 2
        
        # Convert to uncertainty (0-1 scale) with safety checks
        distribution_uncertainty = np.exp(-np.clip(relative_diff, -30, 30))
        
        uncertainty = distribution_uncertainty * (1 + total_var)
        
        # Final check for valid uncertainty value
        if np.isnan(uncertainty) or np.isinf(uncertainty):
            print(f"Warning: Invalid uncertainty value calculated")
            return 0.5
            
        return min(1.0, max(0.0, uncertainty))  # Ensure in [0,1] range
    
    def _ensure_capacity(self, required_size):
        """Ensure that the model can handle at least required_size stimuli"""
        if required_size > self.n_stimuli:
            # Create new weight matrix with expanded size
            new_W1 = np.random.randn(required_size, self.hidden_size) * 0.1
            # Copy over old weights
            new_W1[:self.n_stimuli, :] = self.W1
            self.W1 = new_W1
            
            # Update n_stimuli
            self.n_stimuli = required_size
    
    def copy(self):
        """Create a deep copy of this model"""
        copied = NeuralNetworkModel(self.n_stimuli, self.learning_rate, self.hidden_size, self.dropout_rate)
        copied.W1 = self.W1.copy()
        copied.b1 = self.b1.copy()
        copied.W2 = self.W2.copy()
        copied.b2 = self.b2.copy()
        copied.prediction_history = self.prediction_history.copy()
        return copied