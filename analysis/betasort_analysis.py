import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution

from models import betasort_test
from models.betasort import Betasort

def compare_model_to_one_rat(all_data_df, rat, n_simulations=100, tau=0.01, xi=0.99, threshold=0.75, test=False):
    all_models = {}
    
    # track global stimuli states
    global_U = {}
    global_L = {}
    global_R = {}
    global_N = {}
    
    # store match rates for each day
    match_rates = []
    
    # process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        print(f"\n  Rat {rat} Day {day}", end="", flush=True)
        
        # Extract relevant data
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        #rewards = day_data["correct"].values
        
        # Identify which stimuli are present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1  # +1 because of 0-indexing
        
        # Initialize a new model for this day
        if test:
            model = betasort_test.Betasort(n_stimuli, rat, day)
        else:
            model = Betasort(n_stimuli, rat, day)
        
        # Transfer state from previous days
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
        
        model.uncertainty_history = [model.get_all_stimulus_uncertainties()]
        model.ROC_uncertainty_history = [model.get_all_ROC_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.U_history = [model.U.copy()]
        model.L_history = [model.L.copy()]
        
        # process the trials for today
        participant_choices = np.column_stack((chosen_idx, unchosen_idx))
        n_trials = len(participant_choices)
        matches = np.zeros(n_trials)
        
        for t in range(n_trials):
            if t % 100 == 0 or (n_trials < 100 and t % 10 == 0):
                print(".", end="", flush=True)
            
            chosen_idx, other_idx = participant_choices[t]
            #reward = rewards[t]
            
            # Validate indices (just in case)
            if not (0 <= chosen_idx < n_stimuli) or not (0 <= other_idx < n_stimuli):
                print(f"Day {day}, Trial {t}: Invalid indices - chosen {chosen_idx} unchosen {other_idx}")
                continue

            # run multiple simulations to get choice probability
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx, other_idx])
                model_choices[sim] = model_choice
            
            # see how well the model matches up with real choices
            model_match_rate = np.mean(model_choices == chosen_idx)
            matches[t] = model_match_rate

            # update model based on actual feedback
            reward = 1 if chosen_idx < other_idx else 0
            model.update(chosen_idx, other_idx, reward, model_match_rate, threshold=threshold)
            
            #print(t, chosen_idx, reward)
        
        # calculate cumulative match rate
        cumulative_match_rate = np.mean(matches)
        match_rates.append(cumulative_match_rate)
        
        # store all the stuff for this day
        all_models[day] = model

        # update global states
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
    
    # return the models
    final_day = max(all_models.keys())
    return all_models[final_day], all_models, match_rates

def binomial_analysis_by_session(all_data_df, rat, tau=0.05, xi=0.95, n_simulations=100):
    """Run binomial test to check if model predictions are significantly similar to rat choices"""
    
    all_data_df = all_data_df.sort_values('Day')
    
    session_results = {}
    for day, day_data in all_data_df.groupby('Day'):
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        # initialize model with current parameters
        model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        model_correct = np.zeros(len(chosen_idx))
        rat_correct = np.zeros(len(chosen_idx))
        model_rat_matches = np.zeros(len(chosen_idx))
        
        for t in range(len(chosen_idx)):
            reward = 1 if chosen_idx[t] < unchosen_idx[t] else 0
            sim_correct = np.zeros(n_simulations)
            model_choices = np.zeros(n_simulations)
            
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx[t], unchosen_idx[t]])
                model_choices[sim] = model_choice
                sim_correct[sim] = 1 if model_choice == min(chosen_idx[t], unchosen_idx[t]) else 0
            
            # see how well the model matches up with real choices
            current_model_match_rate = np.mean(model_choices == chosen_idx[t])
            model_rat_matches[t] = current_model_match_rate
            
            model_correct[t] = np.mean(sim_correct)
            rat_correct[t] = reward
            model.update(chosen_idx[t], unchosen_idx[t], reward, current_model_match_rate, threshold=0.6)
        
        n_matches = int(np.sum(model_rat_matches * n_simulations))
        n_trials = len(model_rat_matches) * n_simulations
        chance_match_rate = 0.5  # If there are only 2 choices in each trial
        
        # Test if match rate is significantly higher than chance
        p_value = stats.binomtest(n_matches, n_trials, p=chance_match_rate, alternative='greater')
        
        session_results[day] = {
            'matches': n_matches,
            'trials': n_trials,
            'match_rate': n_matches/n_trials,
            'p_value': p_value,
            'chance_rate': chance_match_rate,
            'significant': p_value.pvalue < 0.05  # Significant if model-rat agreement is BETTER than chance
        }
    
    return session_results

def t_test_model_vs_real_choices(all_data_df, rat, tau=0.05, xi=0.95, n_simulations=100):
    """
    Perform t-tests to compare model choices with real choices across sessions.
    
    Parameters:
        - all_data_df : DataFrame
            - data for all days for a single rat
        - rat : String
            - the rat ID
        - tau, xi : model parameters
        - n_simulations : number of simulations per trial
        
    Returns:
        - session_results : dict
            - contains test results for each day
        - overall_results : dict
            - contains test results across all days
    """
    all_data_df = all_data_df.sort_values('Day')
    
    session_results = {}
    all_model_choices = []
    all_real_choices = []
    
    for day, day_data in all_data_df.groupby('Day'):
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        # Initialize model for this day
        model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        # Track model choices and rat choices for this day
        day_model_choices = []
        day_real_choices = []
        
        for t in range(len(chosen_idx)):
            # Get the rat's choice
            real_choice = chosen_idx[t]
            
            # Simulate model choice multiple times and take the most frequent choice
            model_choices_count = {chosen_idx[t]: 0, unchosen_idx[t]: 0}
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx[t], unchosen_idx[t]])
                model_choices[sim] = model_choice
                model_choices_count[model_choice] += 1
            
            current_model_match_rate = np.mean(model_choices == chosen_idx[t])
            
            # Determine the model's preferred choice
            model_choice = chosen_idx[t] if model_choices_count[chosen_idx[t]] > model_choices_count[unchosen_idx[t]] else unchosen_idx[t]
            
            day_model_choices.append(model_choice)
            day_real_choices.append(real_choice)
            all_model_choices.append(model_choice)
            all_real_choices.append(real_choice)
            
            # Update model based on actual feedback
            reward = 1 if chosen_idx[t] < unchosen_idx[t] else 0
            model.update(chosen_idx[t], unchosen_idx[t], reward, current_model_match_rate, threshold=0.6)
        
        # Convert to arrays
        day_model_choices = np.array(day_model_choices)
        day_real_choices = np.array(day_real_choices)
        
        # make matches the match_model_rate
        # paired 2 way t test in fuure (MAKE SURE PAIRED)
        # Perform paired t-test for this day
        # (comparing whether the model and rat chose the same stimulus)
        matches = (day_model_choices == day_real_choices).astype(int)
        t_stat, p_value = stats.ttest_1samp(matches, 0.5)  # Test if match rate is significantly different from chance
        
        session_results[day] = {
            'n_trials': len(matches),
            'match_rate': np.mean(matches),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Overall test across all days
    all_matches = (np.array(all_model_choices) == np.array(all_real_choices)).astype(int)
    overall_t_stat, overall_p_value = stats.ttest_1samp(all_matches, 0.5)
    
    overall_results = {
        'n_trials': len(all_matches),
        'match_rate': np.mean(all_matches),
        't_statistic': overall_t_stat,
        'p_value': overall_p_value,
        'significant': overall_p_value < 0.05
    }
    
    return session_results, overall_results

def find_optimal_parameters_for_rat(all_data_df, rat, n_simulations=100, xi_values=None, tau_values=None):
    if xi_values is None:
        xi_values_coarse = np.arange(0.75, 0.99 + 0.01, 0.01)
        xi_values_fine = np.arange(0.99, 1.0 + 0.001, 0.001)
        
        # Concatenate arrays and remove any duplicates (0.99 appears in both)
        xi_values = np.unique(np.concatenate([xi_values_coarse, xi_values_fine]))

    if tau_values is None:
        tau_values_fine = np.arange(0.001, 0.01 + 0.001, 0.001)
        tau_values_coarse = np.arange(0.01, 0.25 + 0.01, 0.01)
        
        # Concatenate arrays and remove any duplicates (0.01 appears in both)
        tau_values = np.unique(np.concatenate([tau_values_fine, tau_values_coarse]))
    
    param_performances = {}
    
    # for each parameter combination
    for xi in xi_values:
        for tau in tau_values:
            # start a fresh model with these parameters
            _, _, matches_rate = compare_model_to_one_rat(all_data_df, rat, tau=tau, xi=xi)
            
            param_performances[(xi, tau)] = matches_rate # store rate
    
    # find best parameter combination
    avg_performances = {params: np.mean(day_rates) for params, day_rates in param_performances.items()}
    
    best_params = max(avg_performances, key=avg_performances.get)
    best_xi, best_tau = best_params
    best_performance = avg_performances[best_params]
    
    return best_xi, best_tau, best_performance, param_performances

def find_optimal_threshold(all_data_df, rat, n_simulations=100, xi_values=None, tau_values=None, thresholds=None):
    if xi_values is None:
        xi_values = np.arange(0.9, 1, 0.001)

    if tau_values is None:
        tau_values = np.arange(0, 0.1, 0.001)
    
    if thresholds is None:
        thresholds = np.arange(0.4, 0.7, 0.01)
    
    param_performances = {}
    all_xi = []
    all_tau = []
    all_threshold = []
    all_performances = []
    all_days = []
    
    # list of unique days for labeling later
    unique_days = sorted(all_data_df['Day'].unique())
    num_days = len(unique_days)
    
    # for each parameter combination
    for xi in xi_values:
        for tau in tau_values:
            for threshold in thresholds:
                # start a fresh model with these parameters
                try:
                    _, _, matches_rate = compare_model_to_one_rat(all_data_df, 
                                                                  rat, 
                                                                  tau=tau, 
                                                                  xi=xi, 
                                                                  threshold=threshold)
                    
                    param_performances[(xi, tau, threshold)] = matches_rate # store rate
                    
                    # add list for dataframe
                    for day_idx, day in enumerate(unique_days):
                        all_xi.append(xi)
                        all_tau.append(tau)
                        all_threshold.append(threshold)
                        all_days.append(day)
                        all_performances.append(matches_rate[day_idx])
                
                except Exception as e:
                    print(f"Error with parameters xi={xi}, tau={tau}, threshold={threshold}: {e}")
                    continue
    
    # create dataframe
    results_df = pd.DataFrame({
        'xi': all_xi,
        'tau': all_tau,
        'threshold': all_threshold,
        'day': all_days,
        'match_rate': all_performances
    })
    
    # find best parameter combination
    avg_performances = {}
    for params, day_rates in param_performances.items():
        avg_performances[params] = np.mean(day_rates)
    
    # add a column for average performances across days
    avg_df = results_df.groupby(['xi', 'tau', 'threshold'])['match_rate'].mean().reset_index()
    avg_df['day'] = 'average'
    avg_df = avg_df.rename(columns={'match_rate': 'avg_match_rate'})
    
    # separate dataframe for just the summary
    summary_df = pd.DataFrame({
        'best_xi': [best_xi],
        'best_tau': [best_tau],
        'best_threshold': [best_threshold],
        'best_performance': [best_performance]
    })
    
    best_params = max(avg_performances, key=avg_performances.get)
    best_xi, best_tau, best_threshold = best_params
    best_performance = avg_performances[best_params]
    
    return best_xi, best_tau, best_threshold, best_performance, param_performances, results_df, summary_df

def model_performance_objective(params, *args):
    """Optimization objective function using lightweight model"""
    # Unpack parameters
    xi, tau, threshold = params
    all_data, rat, verbose = args
    
    if verbose:
        #print(f"Evaluating: xi={xi:.6f}, tau={tau:.6f}, threshold={threshold:.6f}", end="", flush=True)
        start_time = time.time()
    
    # Evaluate model with these parameters
    _, _, match_rates = compare_model_to_one_rat(
        all_data, rat,
        tau=tau, xi=xi, threshold=threshold
    )
    
    # Calculate performance
    performance = np.mean(match_rates)
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f" → Performance: {performance:.6f}, Time: {elapsed_time:.2f}s")
    
    # Return negative match rate since we're minimizing
    return -performance

def diff_evolution(all_data, rat, verbose=True, max_iter=100, popsize=25):
    """Optimized differential evolution using lightweight model"""
    if verbose:
        print("\n" + "="*50)
        print(f"Starting differential evolution at {time.strftime('%H:%M:%S')}")
        print(f"Parameter bounds: xi=[0.75, 0.999], tau=[0.001, 0.25], threshold=[0.6, 0.9]")
        print(f"Population size: {popsize}, Max iterations: {max_iter}")
        print("="*50 + "\n")
    
    total_start_time = time.time()
    
    # Store iteration data for tracking
    iterations = [0]
    last_time = [time.time()]
    
    def callback(xk, convergence):
        iterations[0] += 1
        current_time = time.time()
        iter_time = current_time - last_time[0]
        last_time[0] = current_time
        
        best_xi, best_tau, best_threshold = xk
        performance = -model_performance_objective(xk, all_data, rat, False)
        
        if verbose:
            print(f"\nIteration {iterations[0]} completed in {iter_time:.2f}s")
            print(f"  Best parameters: xi={best_xi:.6f}, tau={best_tau:.6f}")
            print(f"  Best performance: {performance:.6f}")
            print(f"  Convergence measure: {convergence:.6f}")
            
            # Estimate remaining time
            elapsed = time.time() - total_start_time
            time_per_iter = elapsed / iterations[0]
            remaining_iters = max_iter - iterations[0]
            est_remaining = time_per_iter * remaining_iters
            print(f"  Estimated remaining time: {est_remaining:.2f}s (~{est_remaining/60:.1f}m)")
        
        return False  # Don't stop the optimization
    
    result = differential_evolution(
        model_performance_objective,
        bounds=[(0.9, 0.999), (0.001, 0.1), (0.6, 0.9)],
        args=(all_data, rat, verbose),
        popsize=popsize,
        maxiter=max_iter,
        mutation=(0.5, 1.5),
        recombination=0.7,
        disp=verbose,
        callback=callback if verbose else None
    )
    
    total_time = time.time() - total_start_time
    
    best_xi, best_tau, best_threshold = result.x
    best_performance = -result.fun
    
    if verbose:
        print("\n" + "="*50)
        print(f"Differential evolution completed at {time.strftime('%H:%M:%S')}")
        print(f"Total optimization time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Total function evaluations: {result.nfev}")
        print(f"Best parameters: xi={best_xi:.6f}, tau={best_tau:.6f}, threshold={best_threshold:.6f}")
        print(f"Best performance: {best_performance:.6f}")
        print("="*50)
    
    return best_xi, best_tau, best_threshold, best_performance

def check_transitive_inference(model, n_simulations=100):
    """
    check over all decision probabilities for each possible choice
    
    Parameters:
    - model : Betasort
        - finished model
    - n_simulations : Int
        - number of simulations for model
    
    Returns:
    - results : {(int, int) : float}
        - chosen_idx, other_idx : % of getting it correct
    """
    
    # check its probabiltiies on what it would choose in transitive inference choices
    results = {}
    
    for chosen_idx in range(0, 4): # max out at 3
        for other_idx in range(chosen_idx + 1, 5):
            # skip same element
            if chosen_idx == other_idx:
                continue
            
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen_idx, other_idx])
                model_choices[sim] = model_choice
            
            # see how well the model matches up with real choices
            model_match_rate = np.mean(model_choices == chosen_idx)
            
            results[(chosen_idx, other_idx)] = model_match_rate
            
    return results

def analyze_vte_uncertainty(all_data_df, rat, tau=0.05, xi=0.95, threshold=0.6, n_simulations=100):
    """Analyze how VTEs correlate with different types of uncertainty

    Args:
        all_data_df (DataFrame): rodent choice data
        vte_data (DataFrame): VTE data:
        
        rat (string): identifier
        tau, xi, threshold (float): model parameters

    Returns:
        pair_vte_data: DataFrame wtih paired VTE & uncertainty
        all_models: Dictionary of models for each day
    """
    pair_vte_data = []
    
    all_models = {}
    global_U, global_L, global_R, global_N = {}, {}, {}, {}
    
    for day, day_data in all_data_df.groupby('Day'):
        # extract relevant data from choice dataset
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        rewards = day_data["correct"].values
        traj_nums = day_data["ID"].values
        vtes = day_data["VTE"].values
        
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        model = Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        # transfer states from previous day
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
            else:
                print(f"stim idx {stim_idx} not found in global updates dict?? {global_U} on day {day}")
            
        # Initialize histories
        model.uncertainty_history = [model.get_all_stimulus_uncertainties()]
        model.ROC_uncertainty_history = [model.get_all_ROC_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.U_history = [model.U.copy()]
        model.L_history = [model.L.copy()]
        
        for t in range(len(chosen_idx)):
            traj_num = traj_nums[t]
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            vte = vtes[t]
            
            # get uncertainty before updates - uncertainty at time of choice
            # individual stimulus uncertainties
            stim1_uncertainty = model.get_uncertainty_stimulus(min(chosen, unchosen))
            stim2_uncertainty = model.get_uncertainty_stimulus(max(chosen, unchosen))
            
            # Get relational uncertainty between the specific pair
            pair_roc_uncertainty = model.get_uncertainty_relation_ROC(min(chosen, unchosen), max(chosen, unchosen))
            
            vte_occurred = 0
            if vte:
                vte_occurred = 1
            
            # Store the pair-specific data
            pair_vte_data.append({
                'day': day,
                'trial_num': traj_num,
                'stim1': min(chosen, unchosen),
                'stim2': max(chosen, unchosen),
                'chosen': chosen,
                'unchosen': unchosen,
                'vte_occurred': vte_occurred,
                'stim1_uncertainty': stim1_uncertainty,
                'stim2_uncertainty': stim2_uncertainty,
                'pair_relational_uncertainty': pair_relational_uncertainty,
                'pair_roc_uncertainty': pair_roc_uncertainty,
                'reward': reward
            })
            
            # run multiple simulations to get choice probability
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen, unchosen])
                model_choices[sim] = model_choice
            
            # see how well the model matches up with real choices
            model_match_rate = np.mean(model_choices == chosen)
            
            # Update the model after recording uncertainty values
            model.update(chosen, unchosen, reward, model_match_rate, threshold=threshold)
        
        # Store model for this day
        all_models[day] = model
        
        # Update global states for the next day
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
    
    # Convert to DataFrame for easier analysis
    pair_vte_df = pd.DataFrame(pair_vte_data)
    
    # Return the data and models
    return pair_vte_df, all_models

def analyze_correlations(pair_vte_df):
    """
    Analyzes correlations between VTE and uncertainty measures for each stimulus pair
    
    Parameters:
        - pair_vte_df: DataFrame with paired VTE and uncertainty values
        
    Returns:
        - results: Dictionary of correlation results by pair
    """
    from scipy.stats import pointbiserialr
    import statsmodels.api as sm
    
    # Identify all unique stimulus pairs
    pair_vte_df['pair'] = pair_vte_df.apply(lambda row: f"{row['stim1']}-{row['stim2']}", axis=1)
    unique_pairs = pair_vte_df['pair'].unique()
    
    # Initialize results dictionary
    results = {
        'overall': {},
        'by_pair': {},
        'by_uncertainty_measure': {}
    }
    
    # Overall correlations (across all pairs)
    uncertainty_measures = [
        'stim1_uncertainty', 
        'stim2_uncertainty', 
        'pair_relational_uncertainty', 
        'pair_roc_uncertainty'
    ]
    
    # Calculate overall correlations
    for measure in uncertainty_measures:
        r, p = pointbiserialr(pair_vte_df['vte_occurred'], pair_vte_df[measure])
        results['overall'][measure] = {'r': r, 'p': p}
    
    # Calculate correlations for each stimulus pair
    for pair in unique_pairs:
        pair_data = pair_vte_df[pair_vte_df['pair'] == pair]
        
        # Skip pairs with very few data points or no VTE variation
        if len(pair_data) < 5 or pair_data['vte_occurred'].nunique() < 2:
            results['by_pair'][pair] = "Insufficient data"
            continue
        
        pair_results = {}
        for measure in uncertainty_measures:
            try:
                r, p = pointbiserialr(pair_data['vte_occurred'], pair_data[measure])
                pair_results[measure] = {'r': r, 'p': p}
            except:
                pair_results[measure] = "Computation failed"
        
        # Logistic regression for this pair
        try:
            X = sm.add_constant(pair_data[uncertainty_measures])
            logit_model = sm.Logit(pair_data['vte_occurred'], X)
            logit_result = logit_model.fit(disp=0)
            
            pair_results['logistic_regression'] = {
                'params': logit_result.params.to_dict(),
                'pvalues': logit_result.pvalues.to_dict(),
                'pseudo_r2': logit_result.prsquared
            }
        except:
            pair_results['logistic_regression'] = "Computation failed"
        
        results['by_pair'][pair] = pair_results
    
    # Analyze by uncertainty measure (which measure best predicts VTE across all pairs)
    for measure in uncertainty_measures:
        # Compare VTE vs non-VTE trials for this measure
        vte_trials = pair_vte_df[pair_vte_df['vte_occurred'] == 1]
        non_vte_trials = pair_vte_df[pair_vte_df['vte_occurred'] == 0]
        
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(
            vte_trials[measure], 
            non_vte_trials[measure], 
            equal_var=False
        )
        
        results['by_uncertainty_measure'][measure] = {
            'mean_vte_trials': vte_trials[measure].mean(),
            'mean_non_vte_trials': non_vte_trials[measure].mean(),
            'difference': vte_trials[measure].mean() - non_vte_trials[measure].mean(),
            't_stat': t_stat,
            'p_value': p_value
        }
    
    return results