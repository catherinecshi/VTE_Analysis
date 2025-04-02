import os

def model_performance_objective(params, *args):
    # unpack parameters
    xi, tau = params
    participant_data, n_stimuli, rat, day, n_simulations = args
    
    # evaluate model with these parameters
    _, match_rate, _ = compare_model_to_rats(
        participant_data, n_stimuli, rat, day,
        tau=tau, xi=xi, n_simulations=n_simulations
    )
    
    # return negative match rate since we're minimizing
    return -match_rate


