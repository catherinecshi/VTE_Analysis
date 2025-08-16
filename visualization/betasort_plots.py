import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats

def plot_stimulus_uncertainty(model, stimulus_labels=None, save=None):
    """
    plot uncertainty over trials
    
    Parameters:
        - model : Betasort
            - fitted model with uncertainty history
        - stimulus_labels : list, optional
            - labels for stimuli
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
        
    # convert history to array
    uncertainty_array = np.array(model.uncertainty_history)
    
    # plot
    plt.figure(figsize=(12, 8))
    
    for i in range(model.n_stimuli):
        plt.plot(uncertainty_array[:, i], label=stimulus_labels[i])
    
    plt.xlabel("Trial")
    plt.ylabel("Uncertainty (Variance)")
    plt.title("Trial-by-Trial Uncertainty from Betasort Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    
def plot_relational_uncertainty(model, stimulus_labels=None, save=None):
    """
    Plot probabilistic-based relational uncertainties between stimuli pairs
    
    Parameters:
        - model : Betasort
            - fitted model with uncertainty history
        - stimulus_labels : list, optional
            - labels for stimuli
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}-{i+1}" for i in range(model.n_stimuli-1)]
        
    # convert history to array
    uncertainty_array = np.array(model.relational_uncertainty_history)
    
    # plot
    plt.figure(figsize=(12, 8))
    
    for i in range(model.n_stimuli-1):
        plt.plot(uncertainty_array[:, i], label=stimulus_labels[i])
    
    plt.xlabel("Trial")
    plt.ylabel("Uncertainty (Variance)")
    plt.title("Trial-by-Trial Probabilistic Uncertainty from Betasort Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_ROC_uncertainty(model, stimulus_labels=None, save=None):
    """
    Plot probabilistic-based relational uncertainties between stimuli pairs
    
    Parameters:
        - model : Betasort
            - fitted model with uncertainty history
        - stimulus_labels : list, optional
            - labels for stimuli
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}-{i+1}" for i in range(model.n_stimuli-1)]
        
    # convert history to array
    uncertainty_array = np.array(model.ROC_uncertainty_history)
    
    # plot
    plt.figure(figsize=(12, 8))
    
    for i in range(model.n_stimuli-1):
        plt.plot(uncertainty_array[:, i], label=stimulus_labels[i])
    
    plt.xlabel("Trial")
    plt.ylabel("Uncertainty (Variance)")
    plt.title("Trial-by-Trial ROC Uncertainty from Betasort Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_relational_uncertainty_history(model, uncertainty_type='both', stimulus_labels=None, figsize=(15, 10), save=None):
    """
    Plot the history of relational uncertainties between stimuli pairs
    
    Parameters:
        - model : Betasort
            - fitted model with prob_uncertainty_relation_history and roc_uncertainty_relation_history
        - uncertainty_type : str
            - 'both', 'prob', or 'roc' to specify which uncertainty measure to plot
        - stimulus_labels : list, optional
            - labels for stimuli
        - figsize : tuple
            - figure size for the plot
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    # Determine which uncertainty types to plot
    plot_prob = uncertainty_type in ['both', 'prob']
    plot_roc = uncertainty_type in ['both', 'roc']
    
    # Create subplot layout based on what we're plotting
    if plot_prob and plot_roc:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        prob_ax, roc_ax = axes
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if plot_prob:
            prob_ax = ax
        else:
            roc_ax = ax
    
    # Plot probability-based relational uncertainty
    if plot_prob:
        # Assuming model.prob_uncertainty_relation_history is a dict with keys as (stim1, stim2) tuples
        for pair, uncertainty_history in model.prob_uncertainty_relation_history.items():
            i, j = pair
            label = f"{stimulus_labels[i]} vs {stimulus_labels[j]}"
            prob_ax.plot(uncertainty_history, label=label)
        
        prob_ax.set_ylabel("Uncertainty (Probability)")
        prob_ax.set_title("Probability-Based Relational Uncertainty")
        prob_ax.grid(True, alpha=0.3)
        prob_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Plot ROC-based relational uncertainty
    if plot_roc:
        # Assuming model.roc_uncertainty_relation_history is a dict with keys as (stim1, stim2) tuples
        for pair, uncertainty_history in model.roc_uncertainty_relation_history.items():
            i, j = pair
            label = f"{stimulus_labels[i]} vs {stimulus_labels[j]}"
            roc_ax.plot(uncertainty_history, label=label)
        
        roc_ax.set_ylabel("Uncertainty (ROC)")
        roc_ax.set_title("ROC-Based Relational Uncertainty")
        roc_ax.grid(True, alpha=0.3)
        roc_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set common x-axis label
    if plot_prob and plot_roc:
        axes[-1].set_xlabel("Trial")
    else:
        plt.xlabel("Trial")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    
def plot_positions(model, stimulus_labels=None, save=None):
    """
    plot positions over trials
    
    Parameters:
        - model : Betasort
            -fitted model with position history
        - stimulus_labels : list, optional
            - labels for stimuli
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    plt.figure(figsize=(12, 8))
    position_array = np.array(model.position_history) # convert into array
    
    for i in range(model.n_stimuli):
        plt.plot(position_array[:, i], label=stimulus_labels[i])
        
    plt.xlabel("Trial")
    plt.ylabel("Estimated Position")
    plt.title("Estimated Stimulus Positions from Betasort Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
        
def plot_positions_across_days(all_models, mode='detailed', stimulus_labels=None, figsize=(12, 8), 
                              show_markers=False, save=None):
    """
    Plot positions of all stimuli across days.
    
    Parameters:
        - all_models : dict
            Dictionary where keys are day numbers and values are Betasort model instances
        - mode : str, optional
            'summary' (default): Plot only the final positions for each day
            'detailed': Plot the full evolution of positions across all trials and days
        - stimulus_labels : list, optional
            Labels for stimuli
        - figsize : tuple, optional
            Figure size
        - show_markers : bool, optional
            Whether to show markers at each trial point in detailed mode
        - save : str, optional
            Path to save the figure
            
    Returns:
        - fig, ax : matplotlib figure and axes objects
    """
    # Get sorted days
    days = sorted(all_models.keys())
    
    # Determine the maximum number of stimuli across all days
    max_stimuli = max(model.n_stimuli for model in all_models.values())
    
    # If no labels provided, create default ones
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(max_stimuli)]
    
    # Create a colormap to use different colors for each stimulus
    colors = cm.tab10(np.linspace(0, 1, max_stimuli))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if mode == 'summary':
        # Summary mode: plot only final positions for each day
        for i in range(max_stimuli):
            positions = []
            days_with_stimulus = []
            
            for day in days:
                model = all_models[day]
                if i < model.n_stimuli:
                    # Get the final position for this stimulus on this day
                    final_positions = model.position_history[-1]
                    positions.append(final_positions[i])
                    days_with_stimulus.append(day)
            
            if positions:  # Only plot if this stimulus appears in any day
                label = stimulus_labels[i] if i < len(stimulus_labels) else f"Stimulus {i}"
                ax.plot(days_with_stimulus, positions, 'o-', color=colors[i], label=label, markersize=8)
        
        # Add labels and title
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Estimated Position', fontsize=12)
        ax.set_title('Final Estimated Stimulus Positions by Day', fontsize=14)
        
        # Adjust x-ticks to show all days
        ax.set_xticks(days)
        ax.set_xticklabels([str(day) for day in days])
    
    elif mode == 'detailed':
        # Detailed mode: plot full evolution across all trials and days
        trial_offset = 0
        day_boundaries = [0]  # Start with 0 as the first boundary
        day_midpoints = []
        
        # Keep track of which stimuli we've already seen and labeled
        labeled_stimuli = set()
        
        # Plot positions for each stimulus across all days
        for day in days:
            model = all_models[day]
            n_trials = len(model.position_history)
            
            # Calculate midpoint for day label
            day_midpoints.append(trial_offset + n_trials / 2)
            
            # Get x-axis values for this day (trial numbers + offset)
            x_values = np.arange(trial_offset, trial_offset + n_trials)
            
            # Convert position_history to numpy array for easier indexing
            position_array = np.array(model.position_history)
            
            # Plot each stimulus for this day
            for i in range(model.n_stimuli):
                # Extract positions for this stimulus across all trials
                stimulus_positions = position_array[:, i]
                
                # Only add a label if we haven't seen this stimulus before
                if i not in labeled_stimuli:
                    label = stimulus_labels[i] if i < len(stimulus_labels) else f"Stimulus {i}"
                    labeled_stimuli.add(i)
                else:
                    label = None
                
                # Choose line style based on show_markers flag
                if show_markers:
                    line_style = 'o-'  # Line with circle markers
                    markersize = 4     # Smaller markers for detailed view
                else:
                    line_style = '-'   # Just a line
                    markersize = None  # No markers
                
                ax.plot(x_values, stimulus_positions, line_style, color=colors[i], 
                       label=label, markersize=markersize)
            
            # Update the trial offset for the next day
            trial_offset += n_trials
            day_boundaries.append(trial_offset)
        
        # Add vertical lines to separate days
        for boundary in day_boundaries[1:-1]:  # Skip the first and last boundaries
            ax.axvline(x=boundary, color='k', linestyle=':', alpha=0.5)
        
        # Add day labels at the top
        #for i, day in enumerate(days):
            #ax.text(day_midpoints[i], 1.02, f"Day {day}", ha='center', va='bottom', 
                   #transform=ax.get_xaxis_transform())
        
        # Add labels and title
        ax.set_xlabel('Trial (continuous across days)', fontsize=12)
        ax.set_ylabel('Estimated Position', fontsize=12)
        ax.set_title('Evolution of Stimulus Positions Across All Trials and Days', fontsize=14)
        
        # Remove x-ticks for clarity (there would be too many trials)
        ax.set_xticks([])
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'summary' or 'detailed'.")
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend (outside the plot if there are many stimuli)
    if max_stimuli > 5:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    else:
        ax.legend(loc='best')
        plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, ax

def plot_uncertainty_across_days(all_models, uncertainty_type='ROC', mode='detailed', 
                                stimulus_labels=None, figsize=(12, 8), show_markers=False, save=None):
    """
    Plot uncertainty values of all stimulus pairs across days.
    
    Parameters:
        - all_models : dict
            Dictionary where keys are day numbers and values are Betasort model instances
        - uncertainty_type : str, optional
            'ROC' (default): Plot ROC-based uncertainty
            'relational': Plot probability-based relational uncertainty
            'stimulus': Plot individual stimulus uncertainty
        - mode : str, optional
            'summary' (default): Plot only the final uncertainty values for each day
            'detailed': Plot the full evolution of uncertainty across all trials and days
        - stimulus_labels : list, optional
            Labels for stimuli
        - figsize : tuple, optional
            Figure size
        - show_markers : bool, optional
            Whether to show markers at each trial point in detailed mode
        - save : str, optional
            Path to save the figure
            
    Returns:
        - fig, ax : matplotlib figure and axes objects
    """
    # Get sorted days
    days = sorted(all_models.keys())
    
    # Determine the maximum number of stimuli across all days
    max_stimuli = max(model.n_stimuli for model in all_models.values())
    
    # If no labels provided, create default ones
    if stimulus_labels is None:
        if uncertainty_type == 'stimulus':
            stimulus_labels = [f"Stimulus {i}" for i in range(max_stimuli)]
        else:
            # For pair-based uncertainties
            stimulus_labels = [f"Pair {i}-{i+1}" for i in range(max_stimuli-1)]
    
    # Create a colormap to use different colors for each stimulus/pair
    if uncertainty_type == 'stimulus':
        colors = cm.tab10(np.linspace(0, 1, max_stimuli))
        num_entities = max_stimuli
    else:
        colors = cm.tab10(np.linspace(0, 1, max_stimuli-1))
        num_entities = max_stimuli-1  # Number of pairs is one less than number of stimuli
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if mode == 'summary':
        # Summary mode: plot only final uncertainty values for each day
        for i in range(num_entities):
            values = []
            days_with_entity = []
            
            for day in days:
                model = all_models[day]
                
                # Skip if this stimulus/pair doesn't exist in this day's model
                if uncertainty_type == 'stimulus' and i >= model.n_stimuli:
                    continue
                elif uncertainty_type != 'stimulus' and i >= model.n_stimuli-1:
                    continue
                
                # Get the appropriate uncertainty history based on type
                if uncertainty_type == 'ROC':
                    history = model.ROC_uncertainty_history
                elif uncertainty_type == 'relational':
                    history = model.relational_uncertainty_history
                elif uncertainty_type == 'stimulus':
                    history = model.uncertainty_history
                else:
                    raise ValueError(f"Invalid uncertainty_type: {uncertainty_type}")
                
                # Get the final uncertainty value for this entity on this day
                final_values = history[-1]
                values.append(final_values[i])
                days_with_entity.append(day)
            
            if values:  # Only plot if this entity appears in any day
                if uncertainty_type == 'stimulus':
                    label = stimulus_labels[i] if i < len(stimulus_labels) else f"Stimulus {i}"
                else:
                    label = stimulus_labels[i] if i < len(stimulus_labels) else f"Pair {i}-{i+1}"
                
                ax.plot(days_with_entity, values, 'o-', color=colors[i], label=label, markersize=8)
        
        # Add labels and title
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Uncertainty', fontsize=12)
        
        if uncertainty_type == 'ROC':
            title = 'ROC Uncertainty Values by Day'
        elif uncertainty_type == 'relational':
            title = 'Relational Uncertainty Values by Day'
        else:
            title = 'Stimulus Uncertainty Values by Day'
            
        ax.set_title(title, fontsize=14)
        
        # Adjust x-ticks to show all days
        ax.set_xticks(days)
        ax.set_xticklabels([str(day) for day in days])
    
    elif mode == 'detailed':
        # Detailed mode: plot full evolution across all trials and days
        trial_offset = 0
        day_boundaries = [0]  # Start with 0 as the first boundary
        day_midpoints = []
        
        # Keep track of which entities we've already labeled
        labeled_entities = set()
        
        # Plot uncertainty values for each entity across all days
        for day in days:
            model = all_models[day]
            
            # Get the appropriate uncertainty history based on type
            if uncertainty_type == 'ROC':
                history = model.ROC_uncertainty_history
            elif uncertainty_type == 'relational':
                history = model.relational_uncertainty_history
            elif uncertainty_type == 'stimulus':
                history = model.uncertainty_history
            
            n_trials = len(history)
            
            # Calculate midpoint for day label
            day_midpoints.append(trial_offset + n_trials / 2)
            
            # Get x-axis values for this day (trial numbers + offset)
            x_values = np.arange(trial_offset, trial_offset + n_trials)
            
            # Convert history to numpy array for easier indexing
            uncertainty_array = np.array(history)
            
            # Plot each entity for this day
            if uncertainty_type == 'stimulus':
                n_entities = model.n_stimuli
            else:
                n_entities = model.n_stimuli - 1
                
            for i in range(n_entities):
                # Extract uncertainty values for this entity across all trials
                entity_values = uncertainty_array[:, i]
                
                # Only add a label if we haven't seen this entity before
                if i not in labeled_entities:
                    if uncertainty_type == 'stimulus':
                        label = stimulus_labels[i] if i < len(stimulus_labels) else f"Stimulus {i}"
                    else:
                        label = stimulus_labels[i] if i < len(stimulus_labels) else f"Pair {i}-{i+1}"
                    labeled_entities.add(i)
                else:
                    label = None
                
                # Choose line style based on show_markers flag
                if show_markers:
                    line_style = 'o-'  # Line with circle markers
                    markersize = 4     # Smaller markers for detailed view
                else:
                    line_style = '-'   # Just a line
                    markersize = None  # No markers
                
                ax.plot(x_values, entity_values, line_style, color=colors[i], 
                       label=label, markersize=markersize)
            
            # Update the trial offset for the next day
            trial_offset += n_trials
            day_boundaries.append(trial_offset)
        
        # Add vertical lines to separate days - now more dotted and visible
        for boundary in day_boundaries[1:-1]:  # Skip the first and last boundaries
            ax.axvline(x=boundary, color='k', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Add day labels at the top
        for i, day in enumerate(days):
            ax.text(day_midpoints[i], 1.02, f"Day {day}", ha='center', va='bottom', 
                   transform=ax.get_xaxis_transform())
        
        # Add labels and title
        ax.set_xlabel('Trial (continuous across days)', fontsize=12)
        ax.set_ylabel('Uncertainty', fontsize=12)
        
        if uncertainty_type == 'ROC':
            title = 'Evolution of ROC Uncertainty Values Across All Trials and Days'
        elif uncertainty_type == 'relational':
            title = 'Evolution of Relational Uncertainty Values Across All Trials and Days'
        else:
            title = 'Evolution of Stimulus Uncertainty Values Across All Trials and Days'
            
        ax.set_title(title, fontsize=14)
        
        # Remove x-ticks for clarity (there would be too many trials)
        ax.set_xticks([])
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'summary' or 'detailed'.")
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend (outside the plot if there are many entities)
    if num_entities > 5:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    else:
        ax.legend(loc='best')
        plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, ax
    
def plot_beta_distributions(model, x_resolution=1000, stimulus_labels=None, figsize=(12, 8), save=None):
    """
    plot beta distributions based on upper and lower parameters at the end
    
    Parameters:
        - model : Betasort
            - fitted model with U & L parameters
        - x_resolution : int, optional
            - higher gives smoother curves
        - stimulus_labels : list, optional
        - figsize : tuple, optional
    
    Returns:
        - fig, ax : matplotlib figure and axes objects
    """
    
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(0, 1, x_resolution)
    
    # maximum density value to scale the plot
    max_density = 0
    
    # for each stimulus
    for i in range(model.n_stimuli):
        # get parameters for beta distribution
        a = model.U[i] + 1
        b = model.L[i] + 1
        
        # create beta distribution
        beta_dist = stats.beta(a, b)
        
        # calculate probability density
        density = beta_dist.pdf(x)
        
        # track maximum density for scaling
        if max(density) > max_density:
            max_density = max(density)
        
        # plot
        ax.plot(x, density, label=f"{stimulus_labels[i]} (alpha={a:.2f}, beta={b:.2f})")
        
        # calculate and mark the mean or expected position
        mean_pos = a / (a + b)
        mean_density = beta_dist.pdf(mean_pos)
        ax.scatter([mean_pos], [mean_density], marker='o', s=50,
                   edgecolor='black', linewidth=1.5, zorder=5)
        
    # add shade
    for i in range(model.n_stimuli):
        a = model.U[i] + 1
        b = model.L[i] + 1
        beta_dist = stats.beta(a, b)
        density = beta_dist.pdf(x)
        ax.fill_between(x, density, alpha=0.1)
    
    # labels
    ax.set_xlabel("Position value", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Beta Distribution for Each Stimulus", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # legend
    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_boundaries_history(model, stimulus_labels=None, save=None):
    """
    plots thei hsitory of the upper and lower parameters
    
    Parameters:
        - model : Betasort
        - stimulus_labels : list, optional
            - labels of stimuli to plot, defaults to all
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    # plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # number of trials
    n_trials = len(model.U_history)
    trials = range(n_trials)
    
    for i, label in enumerate(stimulus_labels):
        ax.plot(trials, [u[i] for u in model.U_history],
                label=f"U - {label}", linestyle="-")
        ax.plot(trials, [l[i] for l in model.L_history],
                label=f"L - {label}", linestyle="--")

        # add final values annotation
        final_U = model.U_history[-1][i]
        final_L = model.L_history[-1][i]
        ax.annotate(f"U{i} = {final_U:.2f}", 
                    xy=(n_trials-1, final_U), xytext=(n_trials-10, final_U*1.05),
                    arrowprops=dict(arrowstyle="->"))
        ax.annotate(f"L{i} = {final_L:.2f}", 
                    xy=(n_trials-1, final_L), xytext=(n_trials-10, final_L*0.95),
                    arrowprops=dict(arrowstyle="->"))
    
    # labels
    ax.set_xlabel("Trial")
    ax.set_ylabel("Parameter Value")
    ax.set_title("History of U and L Parameters")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_rewards_history(model, stimulus_labels=None, save=None):
    """
    plots thei hsitory of the upper and lower parameters
    
    Parameters:
        - model : Betasort
        - stimulus_labels : list, optional
            - labels of stimuli to plot, defaults to all
    """
    if stimulus_labels is None:
        stimulus_labels = [f"Stimulus {i}" for i in range(model.n_stimuli)]
    
    # plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # number of trials
    n_trials = len(model.R_history)
    trials = range(n_trials)
    
    for i, label in enumerate(stimulus_labels):
        ax.plot(trials, [r[i] for r in model.R_history],
                label=f"R - {label}", linestyle="-")
        ax.plot(trials, [n[i] for n in model.N_history],
                label=f"N - {label}", linestyle="--")

        # add final values annotation
        final_R = model.R_history[-1][i]
        final_N = model.N_history[-1][i]
        ax.annotate(f"R{i} = {final_R:.2f}", 
                    xy=(n_trials-1, final_R), xytext=(n_trials-10, final_R*1.05),
                    arrowprops=dict(arrowstyle="->"))
        ax.annotate(f"N{i} = {final_N:.2f}", 
                    xy=(n_trials-1, final_N), xytext=(n_trials-10, final_N*0.95),
                    arrowprops=dict(arrowstyle="->"))
    
    # labels
    ax.set_xlabel("Trial")
    ax.set_ylabel("Parameter Value")
    ax.set_title("History of R and N Parameters")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_ROC_uncertainty_across_days(all_models, mode='detailed', pair_labels=None, figsize=(12, 8), 
                                    show_markers=False, save=None):
    """
    Plot ROC uncertainty between adjacent stimulus pairs across days.
    
    Parameters:
        - all_models : dict
            Dictionary where keys are day numbers and values are Betasort model instances
        - mode : str, optional
            'summary': Plot only the final ROC uncertainty values for each day
            'detailed': Plot the full evolution of ROC uncertainty across all trials and days
        - pair_labels : list, optional
            Labels for stimulus pairs (e.g., ['A-B', 'B-C', 'C-D', 'D-E'])
        - figsize : tuple, optional
            Figure size
        - show_markers : bool, optional
            Whether to show markers at each trial point in detailed mode
        - save : str, optional
            Path to save the figure
            
    Returns:
        - fig, ax : matplotlib figure and axes objects
    """
    # Get sorted days
    days = sorted(all_models.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine the maximum number of pairs across all days
    max_pairs = max(model.n_stimuli - 1 for model in all_models.values())
    
    # If no labels provided, create default ones
    if pair_labels is None:
        pair_labels = [f"Pair {i}-{i+1}" for i in range(max_pairs)]
    
    # Create a colormap to use different colors for each pair
    colors = cm.tab10(np.linspace(0, 1, max_pairs))
    
    if mode == 'summary':
        # Summary mode: plot only final ROC uncertainty values for each day
        for i in range(max_pairs):
            values = []
            days_with_pair = []
            
            for day in days:
                model = all_models[day]
                if i < model.n_stimuli - 1:
                    # Get the final ROC uncertainty for this pair on this day
                    final_values = model.ROC_uncertainty_history[-1]
                    values.append(final_values[i])
                    days_with_pair.append(day)
            
            if values:  # Only plot if this pair appears in any day
                label = pair_labels[i] if i < len(pair_labels) else f"Pair {i}-{i+1}"
                ax.plot(days_with_pair, values, 'o-', color=colors[i], label=label, markersize=8)
        
        # Add labels and title
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('ROC Uncertainty', fontsize=12)
        ax.set_title('Final ROC Uncertainty Between Stimulus Pairs by Day', fontsize=14)
        
        # Adjust x-ticks to show all days
        ax.set_xticks(days)
        ax.set_xticklabels([str(day) for day in days])
    
    elif mode == 'detailed':
        # Detailed mode: plot full evolution across all trials and days
        trial_offset = 0
        day_boundaries = [0]  # Start with 0 as the first boundary
        day_midpoints = []
        
        # Keep track of which pairs we've already labeled
        labeled_pairs = set()
        
        # Plot ROC uncertainty for each pair across all days
        for day in days:
            model = all_models[day]
            
            # Get ROC uncertainty history for this day
            roc_history = model.ROC_uncertainty_history
            n_trials = len(roc_history)
            
            # Calculate midpoint for day label
            day_midpoints.append(trial_offset + n_trials / 2)
            
            # Get x-axis values for this day (trial numbers + offset)
            x_values = np.arange(trial_offset, trial_offset + n_trials)
            
            # Convert ROC_uncertainty_history to numpy array for easier indexing
            uncertainty_array = np.array(roc_history)
            
            # Plot each pair for this day
            for i in range(model.n_stimuli - 1):
                # Extract ROC uncertainty for this pair across all trials
                pair_uncertainty = uncertainty_array[:, i]
                
                # Only add a label if we haven't seen this pair before
                if i not in labeled_pairs:
                    label = pair_labels[i] if i < len(pair_labels) else f"Pair {i}-{i+1}"
                    labeled_pairs.add(i)
                else:
                    label = None
                
                # Choose line style based on show_markers flag
                if show_markers:
                    line_style = 'o-'  # Line with circle markers
                    markersize = 4     # Smaller markers for detailed view
                else:
                    line_style = '-'   # Just a line
                    markersize = None  # No markers
                
                ax.plot(x_values, pair_uncertainty, line_style, color=colors[i], 
                       label=label, markersize=markersize, linewidth=2)
            
            # Update the trial offset for the next day
            trial_offset += n_trials
            day_boundaries.append(trial_offset)
        
        # Add vertical lines to separate days
        for boundary in day_boundaries[1:-1]:  # Skip the first and last boundaries
            ax.axvline(x=boundary, color='k', linestyle=':', alpha=0.7, linewidth=1.5)
        
        # Add day labels at the top
        for i, day in enumerate(days):
            ax.text(day_midpoints[i], 1.02, f"Day {day}", ha='center', va='bottom', 
                   transform=ax.get_xaxis_transform(), fontsize=10)
        
        # Add labels and title
        ax.set_xlabel('Trial (continuous across days)', fontsize=12)
        ax.set_ylabel('ROC Uncertainty', fontsize=12)
        ax.set_title('Evolution of ROC Uncertainty Between Stimulus Pairs Across All Trials and Days', fontsize=14)
        
        # Remove x-ticks for clarity (there would be too many trials)
        ax.set_xticks([])
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'summary' or 'detailed'.")
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend (outside the plot if there are many pairs)
    if max_pairs > 5:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    else:
        ax.legend(loc='best')
        plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    
    return fig, ax

def plot_match_rates(matches, window_size=10, save=None):
    """
    plot trial-by-trial and moving average match rates
    
    Parameters:
        - matches : array
            - trial by trial match rate
        - window_size : int
            - size of the moving average window
    """
    
    n_trials = len(matches)
    trials = np.arange(1, n_trials + 1)
    
    # calculate moving average
    moving_avg = np.convolve(matches, np.ones(window_size)/window_size, mode='valid')
    moving_avg_trials = np.arange(window_size, n_trials + 1)
    
    # plot
    plt.figure(figsize=(12, 6))
    plt.scatter(trials, matches, alpha=0.5, label="Trial-by-trial match rate")
    
    # plot moving average
    plt.plot(moving_avg_trials, moving_avg, 'r-', linewidth=2,
             label=f"Moving average (window={window_size})")

    # plot chance level
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Chance Level')
    
    # plot cumulative average
    cumulative_avg = np.cumsum(matches) / np.arange(1, len(matches) + 1)
    plt.plot(trials, cumulative_avg, 'g-', linewidth=2, label='Cumulative Average')
    
    # labels
    plt.xlabel("Trials")
    plt.ylabel("Match Rate")
    plt.title("Betasort Model Choices vs Real Rodent Choices")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def parameter_performance_heatmap(param_performances, save=None):
    plt.figure(figsize=(10, 8))
    xi_list = sorted(set(p[0] for p in param_performances.keys()))
    tau_list = sorted(set(p[1] for p in param_performances.keys()))
    
    # performance matrix for heatmap
    performance_matrix = np.zeros((len(xi_list), len(tau_list)))
    for i, xi in enumerate(xi_list):
        for j, tau in enumerate(tau_list):
            performance_matrix[i, j] = param_performances.get((xi, tau), 0)
    
    # plot heatmap
    plt.imshow(performance_matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Match Rate')
    plt.xlabel('Tau (noise parameter)')
    plt.ylabel('Xi (recall parameter)')
    plt.title('Parameter Performance Heatmap')
    
    # set tick labels
    plt.xticks(range(len(tau_list)), [f"{t:.3f}" for t in tau_list])
    plt.yticks(range(len(xi_list)), [f"{x:.3f}" for x in xi_list])
    
    # text annotations with values
    for i, xi in enumerate(xi_list):
        for j, tau in enumerate(tau_list):
            plt.text(j, i, f"{performance_matrix[i, j]:.3f}",
                     ha="center", va="center", color="w" if performance_matrix[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def parameter_performance_heatmap_with_threshold(param_performances, title=None, fixed_param=None, fixed_value=None, save=None):
    """
    Create a heatmap of parameter performances
    
    Parameters:
        - param_performances: Dictionary of performance metrics
        - title: Optional title for the plot
        - fixed_param: Which parameter to fix (0 for xi, 1 for tau, 2 for threshold)
        - fixed_value: The value to fix the parameter at
    
    Returns:
        - fig, ax: Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract parameter values
    params_structure = list(param_performances.keys())[0]
    
    if len(params_structure) == 3:  # (xi, tau, threshold)
        # We need to fix one parameter to visualize in 2D
        if fixed_param is None:
            fixed_param = 2  # Default to fixing threshold
            
        # Get unique values for each parameter
        xi_values = sorted(set(p[0] for p in param_performances.keys()))
        tau_values = sorted(set(p[1] for p in param_performances.keys()))
        threshold_values = sorted(set(p[2] for p in param_performances.keys()))
        
        if fixed_param == 0:  # Fix xi
            if fixed_value is None:
                fixed_value = xi_values[len(xi_values) // 2]  # Use middle value
            
            # Filter for fixed xi
            filtered_params = {(tau, thresh): perf 
                              for (xi, tau, thresh), perf in param_performances.items() 
                              if xi == fixed_value}
            
            # Use tau and threshold as axes
            x_label = 'Threshold'
            y_label = 'Tau (noise parameter)'
            x_values = threshold_values
            y_values = tau_values
            
            if title is None:
                title = f'Parameter Performance (Xi = {fixed_value:.3f})'
                
        elif fixed_param == 1:  # Fix tau
            if fixed_value is None:
                fixed_value = tau_values[len(tau_values) // 2]
                
            # Filter for fixed tau
            filtered_params = {(xi, thresh): perf 
                              for (xi, tau, thresh), perf in param_performances.items() 
                              if tau == fixed_value}
            
            # Use xi and threshold as axes
            x_label = 'Threshold'
            y_label = 'Xi (recall parameter)'
            x_values = threshold_values
            y_values = xi_values
            
            if title is None:
                title = f'Parameter Performance (Tau = {fixed_value:.3f})'
                
        else:  # Fix threshold (default)
            if fixed_value is None:
                fixed_value = threshold_values[len(threshold_values) // 2]
                
            # Filter for fixed threshold
            filtered_params = {(xi, tau): perf 
                              for (xi, tau, thresh), perf in param_performances.items() 
                              if thresh == fixed_value}
            
            # Use xi and tau as axes
            x_label = 'Tau (noise parameter)'
            y_label = 'Xi (recall parameter)'
            x_values = tau_values
            y_values = xi_values
            
            if title is None:
                title = f'Parameter Performance (Threshold = {fixed_value:.3f})'
    else:
        # Just (xi, tau) keys
        filtered_params = param_performances
        x_label = 'Tau (noise parameter)'
        y_label = 'Xi (recall parameter)'
        x_values = sorted(set(p[1] for p in param_performances.keys()))
        y_values = sorted(set(p[0] for p in param_performances.keys()))
        
        if title is None:
            title = 'Parameter Performance Heatmap'
    
    # Create performance matrix for heatmap
    performance_matrix = np.zeros((len(y_values), len(x_values)))
    
    # Fill the performance matrix
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            key = (y_val, x_val)
            if key in filtered_params:
                perf_value = filtered_params[key]
                if isinstance(perf_value, (list, np.ndarray)):
                    performance_matrix[i, j] = np.mean(perf_value)
                else:
                    performance_matrix[i, j] = perf_value
    
    # Plot heatmap
    im = ax.imshow(performance_matrix, interpolation='nearest', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Match Rate')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Set tick labels (limit number of ticks for readability)
    max_ticks = 10
    xtick_indices = np.linspace(0, len(x_values)-1, min(max_ticks, len(x_values))).astype(int)
    ytick_indices = np.linspace(0, len(y_values)-1, min(max_ticks, len(y_values))).astype(int)
    
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels([f"{x_values[i]:.3f}" for i in xtick_indices])
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels([f"{y_values[i]:.3f}" for i in ytick_indices])
    
    # Add text annotations if there aren't too many cells
    if len(y_values) * len(x_values) <= 100:
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                ax.text(j, i, f"{performance_matrix[i, j]:.3f}",
                        ha="center", va="center", 
                        color="w" if performance_matrix[i, j] < 0.7 else "black")
    
    return fig, ax

def plot_best_parameters(best_model):
    plot_positions(best_model)
    #plot_uncertainty(best_model)
    plot_beta_distributions(best_model)
    
def plot_vte_uncertainty(pair_vte_df, results, save=None):
    """
    Creates visualizations showing the relationship between VTE and pair-specific uncertainty
    
    Parameters:
        - pair_vte_df: DataFrame with VTE and uncertainty data
        - results: Analysis results from analyze_pair_specific_correlations
        - save: Directory to save plots (optional)
    """
    # 1. Boxplots of uncertainty by VTE for each pair
    unique_pairs = pair_vte_df['pair'].unique()
    
    # Determine plot grid dimensions
    n_pairs = len(unique_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    # For each uncertainty measure
    for measure in ['stim1_uncertainty', 'stim2_uncertainty', 'pair_relational_uncertainty', 'pair_roc_uncertainty']:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        fig.suptitle(f'{measure} by VTE Status for Each Stimulus Pair', fontsize=16)
        
        # Flatten axes array if necessary
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Create boxplot for each pair
        for i, pair in enumerate(unique_pairs):
            pair_data = pair_vte_df[pair_vte_df['pair'] == pair]
            
            if i < len(axes):
                if len(pair_data) > 0 and pair_data['vte_occurred'].nunique() > 1:
                    sns.boxplot(x='vte_occurred', y=measure, data=pair_data, ax=axes[i])
                    
                    # Add correlation info if available
                    if pair in results['by_pair'] and measure in results['by_pair'][pair]:
                        if isinstance(results['by_pair'][pair][measure], dict):
                            r = results['by_pair'][pair][measure]['r']
                            p = results['by_pair'][pair][measure]['p']
                            axes[i].set_title(f'Pair {pair}: r={r:.2f}, p={p:.3f}')
                        else:
                            axes[i].set_title(f'Pair {pair}')
                    else:
                        axes[i].set_title(f'Pair {pair}')
                else:
                    axes[i].set_visible(False)
                    
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for the suptitle
        
        if save:
            save_path = save / f"vte_{measure}_by_pair.png"
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    # 2. Barplot showing correlation strength for each pair
    # Extract correlation values for each pair
    pairs = []
    correlation_values = []
    p_values = []
    measures = []
    
    for pair, pair_results in results['by_pair'].items():
        if isinstance(pair_results, dict):
            for measure in ['stim1_uncertainty', 'stim2_uncertainty', 'pair_relational_uncertainty', 'pair_roc_uncertainty']:
                if measure in pair_results and isinstance(pair_results[measure], dict):
                    pairs.append(pair)
                    correlation_values.append(pair_results[measure]['r'])
                    p_values.append(pair_results[measure]['p'])
                    measures.append(measure)
    
    if correlation_values:  # Only proceed if we have valid correlations
        # Create DataFrame for plotting
        corr_df = pd.DataFrame({
            'Pair': pairs,
            'Correlation': correlation_values,
            'P-value': p_values,
            'Measure': measures,
            'Significant': [p < 0.05 for p in p_values]
        })
        
        # Plot barplot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Pair', y='Correlation', hue='Measure', data=corr_df)
        
        # Add significance markers
        for i, (_, row) in enumerate(corr_df.iterrows()):
            if row['Significant']:
                plt.text(i, row['Correlation'] + 0.02, '*', ha='center', va='center', fontsize=12)
        
        plt.title('VTE-Uncertainty Correlation by Stimulus Pair')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            save_path = save / "vte_correlation_by_pair.png"
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    # 3. Heatmap of uncertainty by day and pair for VTE trials
    # Calculate mean uncertainty for each day-pair combination
    heatmap_data = pair_vte_df.pivot_table(
        index='day', 
        columns='pair',
        values='pair_relational_uncertainty',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f")
    plt.title('Mean Relational Uncertainty by Day and Stimulus Pair')
    plt.tight_layout()
    
    if save:
        save_path = save / "uncertainty_heatmap_by_day_pair.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
    # 4. Line plot of VTE occurrence rate and mean uncertainty over days
    day_summary = pair_vte_df.groupby('day').agg({
        'vte_occurred': 'mean',
        'stim1_uncertainty': 'mean',
        'stim2_uncertainty': 'mean',
        'pair_relational_uncertainty': 'mean',
        'pair_roc_uncertainty': 'mean'
    }).reset_index()
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Day')
    ax1.set_ylabel('VTE Rate', color=color)
    ax1.plot(day_summary['day'], day_summary['vte_occurred'], 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Uncertainty', color=color)
    ax2.plot(day_summary['day'], day_summary['pair_relational_uncertainty'], 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('VTE Rate and Mean Uncertainty Over Days')
    fig.tight_layout()
    
    if save:
        save_path = save / "vte_uncertainty_over_days.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_adjacent_pair_comparison(pair_names, rat_rates, pre_model_rates, post_model_rates, day=None, save=None):
    """
    Plot comparison of rat performance vs model performance (pre and post update) for adjacent pairs
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.arange(len(pair_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, rat_rates, width, label='Actual Rat Performance', color='blue', alpha=0.7)
    bars2 = ax.bar(x, pre_model_rates, width, label='Pre-Update Model', color='orange', alpha=0.7)
    bars3 = ax.bar(x + width, post_model_rates, width, label='Post-Update Model', color='green', alpha=0.7)
    
    ax.set_xlabel('Adjacent Stimulus Pairs')
    ax.set_ylabel('Correct Choice Rate')
    ax.set_title(f'Adjacent Pair Performance Comparison - Day {day}' if day else 'Adjacent Pair Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_aggregated_adjacent_pair_comparison(pair_names, rat_rates, pre_model_rates, post_model_rates, rat_counts, total_rats=None, save=None):
    """
    Plot aggregated comparison of rat performance vs model performance across all rats
    """
    
    x = np.arange(len(pair_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width, rat_rates, width, label='Average Rat Performance', color='blue', alpha=0.7)
    bars2 = ax.bar(x, pre_model_rates, width, label='Average Pre-Update Model', color='orange', alpha=0.7)
    bars3 = ax.bar(x + width, post_model_rates, width, label='Average Post-Update Model', color='green', alpha=0.7)
    
    ax.set_xlabel('Adjacent Stimulus Pairs', fontsize=12)
    ax.set_ylabel('Correct Choice Rate', fontsize=12)
    title = f'Adjacent Pair Performance Comparison - Averaged Across All Rats (n={total_rats})' if total_rats else 'Adjacent Pair Performance Comparison - Averaged Across All Rats'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_value_labels(bars1, rat_rates)
    add_value_labels(bars2, pre_model_rates)
    add_value_labels(bars3, post_model_rates)
    
    # Add sample size information
    for i, (x_pos, count) in enumerate(zip(x, rat_counts)):
        ax.text(x_pos, -0.08, f'n={count}', ha='center', va='top', 
                transform=ax.get_xaxis_transform(), fontsize=9, style='italic')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved aggregated plot to {save}")
    else:
        plt.show()
    
    return fig

def plot_post_model_vs_rat_comparison(pair_names, rat_rates, post_model_rates, rat_counts, total_rats=None, save=None):
    """
    Plot comparison of rat performance vs post-update model performance only
    """
    
    x = np.arange(len(pair_names))
    width = 0.35  # Wider bars since we only have two categories
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, rat_rates, width, label='Rat Performance', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, post_model_rates, width, label='Model Performance', color='green', alpha=0.7)
    
    ax.set_xlabel('Adjacent Stimulus Pairs', fontsize=12)
    ax.set_ylabel('Correct Choice Rate', fontsize=12)
    title = f'Rat vs Model Performance - Averaged Across All Rats (n={total_rats})' if total_rats else 'Rat vs Model Performance - Averaged Across All Rats'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1, rat_rates)
    add_value_labels(bars2, post_model_rates)
    
    # Add sample size information
    for i, (x_pos, count) in enumerate(zip(x, rat_counts)):
        ax.text(x_pos, -0.08, f'n={count}', ha='center', va='top', 
                transform=ax.get_xaxis_transform(), fontsize=9, style='italic')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add correlation information as text
    correlation = np.corrcoef(rat_rates, post_model_rates)[0, 1]
    ax.text(0.02, 0.98, f'Correlation: r = {correlation:.3f}', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved post-model vs rat comparison plot to {save}")
    else:
        plt.show()
    
    return fig

def plot_vte_percentage_comparison(pair_names, rat_vte_percentages, model_vte_percentages, rat_counts, total_rats=None, save=None):
    """
    Plot comparison of VTE percentages between rats and model predictions by trial type
    
    Parameters:
    -----------
    pair_names : list
        Names of stimulus pairs (e.g., ['AB', 'BC', 'CD', 'DE'])
    rat_vte_percentages : list
        Percentage of trials with VTEs for actual rats
    model_vte_percentages : list
        Percentage of trials with VTEs predicted by model
    rat_counts : list
        Number of rats contributing to each pair
    total_rats : int, optional
        Total number of rats in analysis
    save : str, optional
        Path to save the figure
    """
    x = np.arange(len(pair_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, rat_vte_percentages, width, 
                   label='Rat VTE Percentage', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, model_vte_percentages, width, 
                   label='Model VTE Percentage', color='green', alpha=0.7)
    
    ax.set_xlabel('Trial Types', fontsize=12)
    ax.set_ylabel('VTE Percentage', fontsize=12)
    title = f'VTE Percentage Comparison: Rat vs Model - Averaged Across All Rats (n={total_rats})' if total_rats else 'VTE Percentage Comparison: Rat vs Model'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1, rat_vte_percentages)
    add_value_labels(bars2, model_vte_percentages)
    
    # Add sample size information
    for i, (x_pos, count) in enumerate(zip(x, rat_counts)):
        ax.text(x_pos, -0.08, f'n={count}', ha='center', va='top', 
                transform=ax.get_xaxis_transform(), fontsize=9, style='italic')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add correlation information as text
    correlation = np.corrcoef(rat_vte_percentages, model_vte_percentages)[0, 1]
    ax.text(0.02, 0.98, f'Correlation: r = {correlation:.3f}', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved VTE percentage comparison plot to {save}")
    else:
        plt.show()
    
    return fig

def plot_vte_match_rate_by_trial_type(pair_names, vte_match_rates, rat_counts, total_rats=None, save=None):
    """
    Plot match rate between model VTE predictions and actual rat VTEs by trial type
    
    Parameters:
    -----------
    pair_names : list
        Names of stimulus pairs (e.g., ['AB', 'BC', 'CD', 'DE'])
    vte_match_rates : list
        Match rate between model and rat VTEs for each trial type
    rat_counts : list
        Number of rats contributing to each pair
    total_rats : int, optional
        Total number of rats in analysis
    save : str, optional
        Path to save the figure
    """
    x = np.arange(len(pair_names))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(x, vte_match_rates, color='purple', alpha=0.7, 
                  label='VTE Match Rate')
    
    ax.set_xlabel('Trial Types', fontsize=12)
    ax.set_ylabel('VTE Match Rate', fontsize=12)
    title = f'Model-Rat VTE Match Rate by Trial Type - Averaged Across All Rats (n={total_rats})' if total_rats else 'Model-Rat VTE Match Rate by Trial Type'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.set_ylim(0, 1)
    
    # Add chance level line
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance Level')
    ax.legend(fontsize=11)
    
    # Add value labels on bars
    for bar, value in zip(bars, vte_match_rates):
        height = bar.get_height()
        ax.annotate(f'{value:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add sample size information
    for i, (x_pos, count) in enumerate(zip(x, rat_counts)):
        ax.text(x_pos, -0.08, f'n={count}', ha='center', va='top', 
                transform=ax.get_xaxis_transform(), fontsize=9, style='italic')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved VTE match rate plot to {save}")
    else:
        plt.show()
    
    return fig
    