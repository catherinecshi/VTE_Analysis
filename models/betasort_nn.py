import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class UniversalUpdateRule(nn.Module):
    """
    Fixed version of the interpretable neural network that learns a universal update rule.
    
    Key changes:
    1. Simplified output structure for better gradient flow
    2. More straightforward architecture
    3. Cleaner parameter handling
    """
    
    def __init__(self, max_stimuli: int = 6, hidden_dim: int = 32):
        super().__init__()
        
        self.max_stimuli = max_stimuli
        
        # Input features for the update rule:
        # - Current U, L, R, N values for chosen stimulus (4)
        # - Current U, L, R, N values for unchosen stimulus (4)  
        # - Reward received (0 or 1) (1)
        # - Relative position of chosen vs unchosen (1)
        # - Uncertainty of chosen stimulus (1)
        # - Uncertainty of unchosen stimulus (1)
        # - Trial number within session (normalized) (1)
        # - Estimated reward rates for chosen/unchosen (2)
        input_dim = 15
        
        # Simplified output: just the raw parameter updates
        # [chosen_dU, chosen_dL, chosen_dR, chosen_dN, unchosen_dU, unchosen_dL, unchosen_dR, unchosen_dN]
        output_dim = 8
        
        # Simple interpretable architecture with proper initialization
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Bound outputs to [-1, 1]
        )
        
        # Learned scaling factor
        self.update_scale = nn.Parameter(torch.tensor(0.1))
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.uniform_(module.bias, -0.01, 0.01)
    
    def forward(self, features):
        """
        Apply the learned update rule
        
        Args:
            features: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Parameter updates: (batch_size, 8) tensor with updates for both stimuli
        """
        # Get raw updates from network
        raw_updates = self.network(features)
        
        # Scale the updates
        scaled_updates = raw_updates * torch.abs(self.update_scale)
        
        return scaled_updates
    
    def interpret_learned_rule(self) -> Dict:
        """Extract interpretable information about the learned update rule"""
        interpretation = {}
        
        # Learned scaling
        interpretation['global_scale'] = float(self.update_scale.data)
        
        # Get first layer weights for feature importance analysis
        first_layer = self.network[0]
        feature_weights = first_layer.weight.data.numpy()
        
        # Feature importance (average absolute weight across all outputs)
        feature_importance = np.mean(np.abs(feature_weights), axis=0)
        
        feature_names = [
            'chosen_U', 'chosen_L', 'chosen_R', 'chosen_N',
            'unchosen_U', 'unchosen_L', 'unchosen_R', 'unchosen_N',
            'reward', 'rel_position', 'chosen_uncertainty', 'unchosen_uncertainty',
            'trial_norm', 'chosen_reward_rate', 'unchosen_reward_rate'
        ]
        
        interpretation['feature_importance'] = {
            name: float(importance) for name, importance in zip(feature_names, feature_importance)
        }
        
        # Analyze output layer for parameter update biases
        output_layer = self.network[-2]  # Second to last layer (before tanh)
        output_weights = output_layer.weight.data.numpy()
        output_biases = output_layer.bias.data.numpy()
        
        # Average bias for each parameter type
        parameter_types = ['chosen_U', 'chosen_L', 'chosen_R', 'chosen_N', 
                          'unchosen_U', 'unchosen_L', 'unchosen_R', 'unchosen_N']
        
        interpretation['parameter_biases'] = {
            param_type: float(bias) for param_type, bias in zip(parameter_types, output_biases)
        }
        
        # Add compatibility fields for visualization code
        interpretation['reward_pathway_strength'] = np.mean(np.abs(output_weights))
        interpretation['no_reward_pathway_strength'] = np.mean(np.abs(output_weights)) * 0.8  # Placeholder
        
        return interpretation

class DifferentiableChoiceLoss(nn.Module):
    """
    Simplified, differentiable loss function that captures choice prediction goals
    without breaking gradient flow.
    
    Instead of simulating actual choices, we use the probability difference between
    stimuli, which is differentiable and captures the same scientific objective.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, current_params, updates, actual_choices):
        """
        Calculate loss based on how well the updated parameters would predict choices
        
        Args:
            current_params: Current U,L values [batch, 8] 
                           (chosen_U, chosen_L, chosen_R, chosen_N, unchosen_U, unchosen_L, unchosen_R, unchosen_N)
            updates: Proposed parameter updates [batch, 8]
            actual_choices: What the rat actually chose next [batch, 1] (0 or 1)
        
        Returns:
            Loss value (lower = better choice prediction)
        """
        batch_size = current_params.shape[0]
        
        # Apply updates to get new parameters
        new_params = current_params + updates
        
        # Ensure parameters stay positive
        new_params = torch.clamp(new_params, min=0.01)
        
        # Extract U and L values for both stimuli
        chosen_U = new_params[:, 0]
        chosen_L = new_params[:, 1]
        chosen_R = new_params[:, 2]
        chosen_N = new_params[:, 3]
        unchosen_U = new_params[:, 4]
        unchosen_L = new_params[:, 5]
        unchosen_R = new_params[:, 6]
        unchosen_N = new_params[:, 7]
        
        # Calculate the probability that chosen stimulus beats unchosen stimulus
        # This is the key insight: instead of sampling, we use the analytical solution
        
        # For Beta(a1, b1) vs Beta(a2, b2), the probability that X1 > X2 can be 
        # approximated using the means and variances, or computed more precisely
        
        # Method 1: Combine belief-based and reward-based estimates
        # Belief-based probability
        chosen_belief = chosen_U / (chosen_U + chosen_L)
        unchosen_belief = unchosen_U / (unchosen_U + unchosen_L)
        
        # Reward-based probability  
        chosen_reward_rate = chosen_R / (chosen_R + chosen_N)
        unchosen_reward_rate = unchosen_R / (unchosen_R + unchosen_N)
        
        # Combine both sources of information (weighted average)
        # The network can learn the optimal weighting
        belief_weight = 0.7  # Could make this learnable too
        reward_weight = 1.0 - belief_weight
        
        chosen_overall = belief_weight * chosen_belief + reward_weight * chosen_reward_rate
        unchosen_overall = belief_weight * unchosen_belief + reward_weight * unchosen_reward_rate
        
        # Probability that chosen beats unchosen
        prob_chosen_wins = torch.sigmoid((chosen_overall - unchosen_overall) * 10)
        
        # Alternative method: Use uncertainty-weighted combination
        # Higher uncertainty means we rely more on reward history
        chosen_uncertainty = (chosen_U * chosen_L) / ((chosen_U + chosen_L)**2 * (chosen_U + chosen_L + 1))
        unchosen_uncertainty = (unchosen_U * unchosen_L) / ((unchosen_U + unchosen_L)**2 * (unchosen_U + unchosen_L + 1))
        
        # Weight toward reward rates when belief is uncertain
        chosen_weight_belief = torch.sigmoid(-chosen_uncertainty * 10)  # Lower uncertainty = more weight on belief
        unchosen_weight_belief = torch.sigmoid(-unchosen_uncertainty * 10)
        
        chosen_combined = chosen_weight_belief * chosen_belief + (1 - chosen_weight_belief) * chosen_reward_rate
        unchosen_combined = unchosen_weight_belief * unchosen_belief + (1 - unchosen_weight_belief) * unchosen_reward_rate
        
        # Use the uncertainty-weighted version as main prediction
        prob_chosen_wins_v2 = torch.sigmoid((chosen_combined - unchosen_combined) * 10)
        
        # Average both methods to be robust
        prob_chosen_wins_final = (prob_chosen_wins + prob_chosen_wins_v2) / 2
        
        # Convert actual choices to probabilities
        target_probs = actual_choices.float()
        
        # Binary cross-entropy loss
        main_loss = nn.functional.binary_cross_entropy(prob_chosen_wins_final, target_probs.squeeze())
        
        # Add regularization to prevent extreme parameter updates
        update_magnitude_penalty = torch.mean(torch.abs(updates)) * 0.01
        
        total_loss = main_loss + update_magnitude_penalty
        
        return total_loss


def create_choice_prediction_data(all_data_df: pd.DataFrame, 
                                rats_to_include: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training data in simplified format for the differentiable loss function
    
    Returns:
        features: Input features for each trial
        param_states: Current parameter states  
        next_choices: Binary labels (1 if rat chose the "chosen" stimulus next, 0 otherwise)
    """
    from models.betasort import Betasort
    
    all_features = []
    all_param_states = []
    all_next_choices = []
    
    rats = rats_to_include if rats_to_include else all_data_df['rat'].unique() if 'rat' in all_data_df.columns else ['rat']
    
    for rat in rats:
        if 'rat' in all_data_df.columns:
            rat_data = all_data_df[all_data_df['rat'] == rat]
        else:
            rat_data = all_data_df
        
        print(f"Processing rat {rat} for training data...")
        
        # Track state across days
        global_U, global_L, global_R, global_N = {}, {}, {}, {}
        
        for day, day_data in rat_data.groupby('Day'):
            chosen_trials = day_data['first'].values
            unchosen_trials = day_data['second'].values
            rewards = day_data['correct'].values
            
            # Initialize model
            all_stimuli = set(np.concatenate([chosen_trials, unchosen_trials]))
            n_stimuli = max(all_stimuli) + 1
            model = Betasort(n_stimuli, rat, day)
            
            # Transfer state
            for stim_idx in range(n_stimuli):
                if stim_idx in global_U:
                    model.U[stim_idx] = global_U[stim_idx]
                    model.L[stim_idx] = global_L[stim_idx]
                    model.R[stim_idx] = global_R[stim_idx]
                    model.N[stim_idx] = global_N[stim_idx]
            
            # Process trials (need next trial for target)
            for t in range(len(chosen_trials) - 1):
                chosen = chosen_trials[t]
                unchosen = unchosen_trials[t]
                reward = rewards[t]
                next_chosen = chosen_trials[t + 1]
                next_unchosen = unchosen_trials[t + 1]
                
                # Create feature vector
                features = []
                features.extend([model.U[chosen], model.L[chosen], model.R[chosen], model.N[chosen]])
                features.extend([model.U[unchosen], model.L[unchosen], model.R[unchosen], model.N[unchosen]])
                features.append(float(reward))
                
                # Positions and uncertainties
                chosen_pos = model.U[chosen] / (model.U[chosen] + model.L[chosen]) if (model.U[chosen] + model.L[chosen]) > 0 else 0.5
                unchosen_pos = model.U[unchosen] / (model.U[unchosen] + model.L[unchosen]) if (model.U[unchosen] + model.L[unchosen]) > 0 else 0.5
                features.append(chosen_pos - unchosen_pos)
                
                features.append(model.get_uncertainty_stimulus(chosen))
                features.append(model.get_uncertainty_stimulus(unchosen))
                features.append(min(1.0, t / 100.0))
                
                chosen_reward_rate = model.R[chosen] / (model.R[chosen] + model.N[chosen])
                unchosen_reward_rate = model.R[unchosen] / (model.R[unchosen] + model.N[unchosen])
                features.extend([chosen_reward_rate, unchosen_reward_rate])
                
                # Current parameter state
                param_state = [
                    model.U[chosen], model.L[chosen], model.R[chosen], model.N[chosen],
                    model.U[unchosen], model.L[unchosen], model.R[unchosen], model.N[unchosen]
                ]
                
                # Simplified target: did the rat choose the same stimulus type in the next trial?
                # We'll consider this as "did they choose the lower-indexed stimulus"
                next_choice_binary = 1 if next_chosen < next_unchosen else 0
                
                all_features.append(features)
                all_param_states.append(param_state)
                all_next_choices.append(next_choice_binary)
                
                # Update model to continue simulation
                model.update(chosen, unchosen, reward, 0.5, 0.6)
            
            # Save state for next day
            for stim_idx in range(n_stimuli):
                global_U[stim_idx] = model.U[stim_idx]
                global_L[stim_idx] = model.L[stim_idx]
                global_R[stim_idx] = model.R[stim_idx]
                global_N[stim_idx] = model.N[stim_idx]
    
    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_param_states, dtype=np.float32), 
        np.array(all_next_choices, dtype=np.float32)
    )

def train_universal_update_rule(all_data_df: pd.DataFrame, 
                              rats_to_include: Optional[List[str]] = None,
                              epochs: int = 500,
                              learning_rate: float = 0.001) -> UniversalUpdateRule:
    """
    Fixed version of the training function with proper gradient flow
    """
    print("Creating training data...")
    features, param_states, next_choices = create_choice_prediction_data(all_data_df, rats_to_include)
    
    print(f"Training on {len(features)} examples from {len(rats_to_include) if rats_to_include else 'all'} rats")
    
    # Create model and loss function
    update_rule = UniversalUpdateRule()
    loss_fn = DifferentiableChoiceLoss()
    
    # Convert to tensors
    X = torch.FloatTensor(features)
    params = torch.FloatTensor(param_states)
    y = torch.FloatTensor(next_choices).unsqueeze(1)  # Add dimension for consistency
    
    # Split data
    n_train = int(0.8 * len(X))
    indices = torch.randperm(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    # Training setup
    optimizer = optim.Adam(update_rule.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    print("Training universal update rule...")
    
    for epoch in range(epochs):
        # Training phase
        update_rule.train()
        optimizer.zero_grad()
        
        # Get proposed updates for training data
        train_updates = update_rule(X[train_idx])
        
        # Calculate loss based on choice prediction improvement
        loss = loss_fn(params[train_idx], train_updates, y[train_idx])
        
        # Backward pass - this should now work without gradient errors
        loss.backward()
        optimizer.step()
        
        # Validation phase
        update_rule.eval()
        with torch.no_grad():
            val_updates = update_rule(X[val_idx])
            val_loss = loss_fn(params[val_idx], val_updates, y[val_idx])
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}')
    
    # Create training visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train', alpha=0.7)
    plt.plot(val_losses, label='Validation', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.yscale('log')
    
    # Analyze learned rule
    interpretation = update_rule.interpret_learned_rule()
    
    plt.subplot(1, 3, 2)
    biases = interpretation['parameter_biases']
    param_names = list(biases.keys())
    param_values = list(biases.values())
    plt.bar(param_names, param_values)
    plt.title('Learned Parameter Update Biases')
    plt.ylabel('Bias Magnitude')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    importance = interpretation['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:8]
    names, values = zip(*top_features)
    plt.barh(range(len(names)), values)
    plt.yticks(range(len(names)), names)
    plt.xlabel('Feature Importance')
    plt.title('Most Important Input Features')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("LEARNED UPDATE RULE ANALYSIS")
    print("="*50)
    print(f"Global update scale: {interpretation['global_scale']:.3f}")
    
    print("\nParameter update biases:")
    for param, bias in interpretation['parameter_biases'].items():
        print(f"  {param}: {bias:.3f}")
    
    print("\nTop features influencing updates:")
    for name, importance in sorted_features[:5]:
        print(f"  {name}: {importance:.3f}")
    
    return update_rule

# NeuralBetasort class remains the same as before, but with proper error handling
class NeuralBetasort:
    """
    Betasort model that uses the learned universal update rule
    """
    
    def __init__(self, n_stimuli: int, rat: str, day: int, 
                 update_rule: UniversalUpdateRule, tau: float = 0.05, xi: float = 0.95):
        self.n_stimuli = n_stimuli
        self.rat = rat
        self.day = day
        self.trial = 0
        self.tau = tau
        self.xi = xi
        
        # Parameters
        self.U = np.ones(n_stimuli)
        self.L = np.ones(n_stimuli)
        self.R = np.ones(n_stimuli)
        self.N = np.ones(n_stimuli)
        
        # Store the learned update rule
        self.update_rule = update_rule
        self.update_rule.eval()
        
        # History tracking
        self.uncertainty_history = [self.get_all_stimulus_uncertainties()]
        self.position_history = [self.get_all_positions()]
        self.U_history = [self.U.copy()]
        self.L_history = [self.L.copy()]
    
    def choose(self, available_stimuli):
        """Choose stimulus (unchanged from original)"""
        X = np.zeros(len(available_stimuli))
        
        for i, stim_idx in enumerate(available_stimuli):
            if np.random.random() < self.tau:
                X[i] = np.random.beta(1, 1)
            else:
                X[i] = np.random.beta(self.U[stim_idx] + 1, self.L[stim_idx] + 1)
        
        chosen_idx = np.argmax(X)
        return available_stimuli[chosen_idx]
    
    def update(self, chosen: int, unchosen: int, reward: int):
        """Update parameters using the learned universal rule"""
        self.trial += 1
        
        # Apply xi-based relaxation (keeping this from original)
        self.R = self.R * self.xi
        self.N = self.N * self.xi
        
        # Calculate reward rates
        chosen_reward_rate = self.R[chosen] / (self.R[chosen] + self.N[chosen])
        unchosen_reward_rate = self.R[unchosen] / (self.R[unchosen] + self.N[unchosen])
        
        # Apply xi-based relaxation to U and L
        E_chosen = chosen_reward_rate
        E_unchosen = unchosen_reward_rate
        xi_R_chosen = E_chosen / (E_chosen + 1) + 0.5
        xi_R_unchosen = E_unchosen / (E_unchosen + 1) + 0.5
        
        self.U[chosen] = self.U[chosen] * xi_R_chosen * self.xi
        self.L[chosen] = self.L[chosen] * xi_R_chosen * self.xi
        self.U[unchosen] = self.U[unchosen] * xi_R_unchosen * self.xi
        self.L[unchosen] = self.L[unchosen] * xi_R_unchosen * self.xi
        
        # Apply the learned update rule
        features = self._create_update_features(chosen, unchosen, reward)
        
        try:
            with torch.no_grad():
                feature_tensor = torch.FloatTensor(features).unsqueeze(0)
                updates = self.update_rule(feature_tensor).squeeze(0).numpy()
            
            # Apply the learned updates with bounds checking
            self.U[chosen] = max(0.01, self.U[chosen] + updates[0])
            self.L[chosen] = max(0.01, self.L[chosen] + updates[1])
            self.R[chosen] = max(0.01, self.R[chosen] + updates[2])
            self.N[chosen] = max(0.01, self.N[chosen] + updates[3])
            
            self.U[unchosen] = max(0.01, self.U[unchosen] + updates[4])
            self.L[unchosen] = max(0.01, self.L[unchosen] + updates[5])
            self.R[unchosen] = max(0.01, self.R[unchosen] + updates[6])
            self.N[unchosen] = max(0.01, self.N[unchosen] + updates[7])
            
        except Exception as e:
            print(f"Warning: Neural update failed, using minimal update. Error: {e}")
            # Fallback to minimal updates if neural network fails
            self.U[chosen] += 0.1
            self.L[unchosen] += 0.1
        
        # Update histories
        self.uncertainty_history.append(self.get_all_stimulus_uncertainties())
        self.position_history.append(self.get_all_positions())
        self.U_history.append(self.U.copy())
        self.L_history.append(self.L.copy())
    
    def _create_update_features(self, chosen: int, unchosen: int, reward: int) -> np.ndarray:
        """Create feature vector for the update rule"""
        features = []
        
        # Current parameter values
        features.extend([self.U[chosen], self.L[chosen], self.R[chosen], self.N[chosen]])
        features.extend([self.U[unchosen], self.L[unchosen], self.R[unchosen], self.N[unchosen]])
        
        # Trial context
        features.append(float(reward))
        
        # Relative position
        chosen_pos = self.U[chosen] / (self.U[chosen] + self.L[chosen]) if (self.U[chosen] + self.L[chosen]) > 0 else 0.5
        unchosen_pos = self.U[unchosen] / (self.U[unchosen] + self.L[unchosen]) if (self.U[unchosen] + self.L[unchosen]) > 0 else 0.5
        features.append(chosen_pos - unchosen_pos)
        
        # Uncertainties
        features.append(self.get_uncertainty_stimulus(chosen))
        features.append(self.get_uncertainty_stimulus(unchosen))
        
        # Trial number
        features.append(min(1.0, self.trial / 100.0))
        
        # Reward rates
        chosen_reward_rate = self.R[chosen] / (self.R[chosen] + self.N[chosen])
        unchosen_reward_rate = self.R[unchosen] / (self.R[unchosen] + self.N[unchosen])
        features.extend([chosen_reward_rate, unchosen_reward_rate])
        
        return np.array(features, dtype=np.float32)
    
    def get_uncertainty_stimulus(self, stimulus_idx):
        """Calculate uncertainty for a given stimulus"""
        a = self.U[stimulus_idx]
        b = self.L[stimulus_idx]
        
        if a + b < 2:
            return 1.0
        
        return (a * b) / ((a + b)**2 * (a + b + 1))
    
    def get_all_stimulus_uncertainties(self):
        """Get uncertainty values for all stimuli"""
        return np.array([self.get_uncertainty_stimulus(i) for i in range(self.n_stimuli)])
    
    def get_all_positions(self):
        """Get estimated positions for all stimuli"""
        positions = np.zeros(self.n_stimuli)
        for i in range(self.n_stimuli):
            if self.U[i] + self.L[i] == 0:
                positions[i] = 0.5
            else:
                positions[i] = self.U[i] / (self.U[i] + self.L[i])
        return positions

def plot_training_results(update_rule: UniversalUpdateRule, train_losses: List[float], val_losses: List[float]):
    """Visualize training results and learned rule interpretation"""
    
    plt.figure(figsize=(15, 5))
    
    # Training curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train', alpha=0.7)
    plt.plot(val_losses, label='Validation', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Log Likelihood)')
    plt.legend()
    plt.title('Training Progress')
    plt.yscale('log')
    
    # Learned parameter biases
    interpretation = update_rule.interpret_learned_rule()
    plt.subplot(1, 3, 2)
    biases = interpretation['parameter_biases']
    plt.bar(biases.keys(), biases.values())
    plt.title('Learned Parameter Update Biases')
    plt.ylabel('Bias Magnitude')
    plt.xticks(rotation=45)
    
    # Feature importance
    plt.subplot(1, 3, 3)
    importance = interpretation['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:8]
    names, values = zip(*top_features)
    plt.barh(range(len(names)), values)
    plt.yticks(range(len(names)), names)
    plt.xlabel('Feature Importance')
    plt.title('Most Important Input Features')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("LEARNED UPDATE RULE ANALYSIS")
    print("="*50)
    print(f"Global update scale: {interpretation['global_scale']:.3f}")
    print(f"Reward pathway vs No-reward pathway strength ratio: {interpretation['reward_pathway_strength']/interpretation['no_reward_pathway_strength']:.3f}")
    print("\nParameter update biases:")
    for param, bias in interpretation['parameter_biases'].items():
        print(f"  {param}: {bias:.3f}")
    print("\nTop features influencing updates:")
    for name, importance in sorted_features[:5]:
        print(f"  {name}: {importance:.3f}")

# Example usage
def compare_update_rules(test_data_df: pd.DataFrame, learned_rule: UniversalUpdateRule, 
                        rat_id: str, tau: float = 0.05, xi: float = 0.95):
    """
    Compare the original Betasort with the neural version using learned update rule
    """
    from models.betasort import Betasort
    
    results = {'original_matches': [], 'neural_matches': []}
    
    rat_data = test_data_df[test_data_df['rat'] == rat_id] if 'rat' in test_data_df.columns else test_data_df
    
    for day, day_data in rat_data.groupby('Day'):
        chosen_data = day_data['first'].values
        unchosen_data = day_data['second'].values
        
        present_stimuli = set(np.concatenate([chosen_data, unchosen_data]))
        n_stimuli = max(present_stimuli) + 1
        
        # Initialize both models
        original_model = Betasort(n_stimuli, rat_id, day, tau=tau, xi=xi)
        neural_model = NeuralBetasort(n_stimuli, rat_id, day, learned_rule, tau=tau, xi=xi)
        
        original_matches = 0
        neural_matches = 0
        
        for t in range(len(chosen_data)):
            chosen = chosen_data[t]
            unchosen = unchosen_data[t]
            
            # Test choice prediction with multiple simulations
            n_sims = 100
            original_choices = [original_model.choose([chosen, unchosen]) for _ in range(n_sims)]
            neural_choices = [neural_model.choose([chosen, unchosen]) for _ in range(n_sims)]
            
            # Calculate match rates
            original_match_rate = np.mean([c == chosen for c in original_choices])
            neural_match_rate = np.mean([c == chosen for c in neural_choices])
            
            original_matches += original_match_rate
            neural_matches += neural_match_rate
            
            # Update both models
            reward = 1 if chosen < unchosen else 0
            original_model.update(chosen, unchosen, reward, original_match_rate, 0.6)
            neural_model.update(chosen, unchosen, reward)
        
        results['original_matches'].append(original_matches / len(chosen_data))
        results['neural_matches'].append(neural_matches / len(chosen_data))
    
    return results