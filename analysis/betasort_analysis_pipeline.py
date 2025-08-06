"""
Betasort Analysis Pipeline - Modular and Configurable

This module provides a clean, modular approach to running comprehensive betasort analysis.
It replaces the unwieldy betasort_pipeline.py with better organization and configurability.

Key features:
- Easy configuration of which rats to include/exclude
- Simple model selection (betasort vs betasort_test)
- Modular functions for each analysis component
- Clear separation of concerns
- Generates all the same outputs as the original pipeline

Usage:
    from analysis.betasort_analysis_pipeline import BetasortPipeline
    
    pipeline = BetasortPipeline(
        rats_to_include=['rat1', 'rat2'],
        rats_to_exclude=['BP09'],
        model_type='betasort_test',
        use_diff_evolution=False
    )
    pipeline.run_full_analysis()
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import matplotlib.pyplot as plt

from config.paths import paths
from config.settings import HIERARCHY_MAPPINGS
from analysis import betasort_analysis
from models import betasort
from models import betasort_test
from models import betasort_OG
from visualization import betasort_plots


def convert_numeric_labels_to_letters(labels):
    """
    Convert numeric labels like '0-1' or '1-3' to letter labels like 'AB' or 'BD'
    using the HIERARCHY_MAPPINGS from settings
    """
    # Create reverse mapping (number -> letter)
    number_to_letter = {v: k for k, v in HIERARCHY_MAPPINGS.items()}
    
    converted_labels = []
    for label in labels:
        if '-' in str(label):
            # Split on hyphen and convert each number
            parts = str(label).split('-')
            try:
                letter_parts = [number_to_letter[int(part)] for part in parts]
                converted_labels.append(''.join(letter_parts))  # Join without hyphen
            except (ValueError, KeyError):
                # If conversion fails, keep original label
                converted_labels.append(str(label).replace('-', ''))
        else:
            # Single number
            try:
                converted_labels.append(number_to_letter[int(label)])
            except (ValueError, KeyError):
                converted_labels.append(str(label))
    
    return converted_labels


class BetasortPipeline:
    """
    Main pipeline class for running comprehensive betasort analysis
    """
    
    def __init__(self, 
                 rats_to_include=None, 
                 rats_to_exclude=None, 
                 model_type='betasort_test',
                 use_diff_evolution=False,
                 n_simulations=100,
                 default_params=None,
                 data_path=None,
                 save_path=None,
                 verbose=True):
        """
        Initialize the analysis pipeline
        
        Parameters:
        -----------
        rats_to_include : list, optional
            Specific rats to include in analysis. If None, includes all rats.
        rats_to_exclude : list, optional
            Specific rats to exclude from analysis (e.g., ['BP09'])
        model_type : str
            Which model to use: 'betasort', 'betasort_test', or 'betasort_OG'
        use_diff_evolution : bool
            Whether to use differential evolution for parameter optimization
        n_simulations : int
            Number of simulations for choice testing
        default_params : dict, optional
            Default parameters if not using differential evolution
            Format: {'xi': 0.95, 'tau': 0.05, 'threshold': 0.8}
        data_path : str, optional
            Path to preprocessed data. Uses paths.preprocessed_data_model if None
        save_path : str, optional
            Path to save results. Uses paths.betasort_data if None
        verbose : bool
            Whether to print progress messages
        """
        self.rats_to_include = rats_to_include
        self.rats_to_exclude = rats_to_exclude or []
        self.model_type = model_type
        self.use_diff_evolution = use_diff_evolution
        self.n_simulations = n_simulations
        self.verbose = verbose
        
        # Default parameters
        self.default_params = default_params or {
            'xi': 0.95, 
            'tau': 0.05, 
            'threshold': 0.8
        }
        
        # Paths
        self.data_path = data_path or paths.preprocessed_data_model
        self.save_path = save_path or paths.betasort_data
        
        # Storage for results
        self.rat_results = {}
        self.aggregated_data = {}
        
        if self.verbose:
            print(f"Initialized Betasort Pipeline:")
            print(f"  Model type: {self.model_type}")
            print(f"  Use differential evolution: {self.use_diff_evolution}")
            print(f"  Number of simulations: {self.n_simulations}")
            print(f"  Data path: {self.data_path}")
            print(f"  Save path: {self.save_path}")
    
    def get_rats_to_process(self):
        """
        Determine which rats to process based on include/exclude lists
        
        Returns:
        --------
        list : List of rat names to process
        """
        all_rats = [rat for rat in os.listdir(self.data_path) 
                   if os.path.isdir(os.path.join(self.data_path, rat))]
        
        # Apply inclusion filter
        if self.rats_to_include is not None:
            rats = [rat for rat in all_rats if rat in self.rats_to_include]
        else:
            rats = all_rats
        
        # Apply exclusion filter
        rats = [rat for rat in rats if rat not in self.rats_to_exclude]
        
        if self.verbose:
            print(f"Processing {len(rats)} rats: {rats}")
            if self.rats_to_exclude:
                print(f"Excluding: {self.rats_to_exclude}")
        
        return rats
    
    def analyze_single_rat(self, rat_name, rat_data):
        """
        Run comprehensive analysis for a single rat
        
        Parameters:
        -----------
        rat_name : str
            Name of the rat
        rat_data : pd.DataFrame
            Preprocessed data for this rat
            
        Returns:
        --------
        dict : Analysis results for this rat
        """
        if self.verbose:
            print(f"Analyzing {rat_name}...")
        
        try:
            # Run the comprehensive analysis (equivalent to analyze_betasort_comprehensive)
            results = self._run_comprehensive_analysis(rat_data, rat_name)
            
            # Save individual rat results
            self._save_rat_results(rat_name, results)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing {rat_name}: {e}")
            return None
    
    def _run_comprehensive_analysis(self, all_data_df, rat):
        """
        Core analysis function - equivalent to analyze_betasort_comprehensive
        """
        results = {}
        
        # Step 1: Parameter optimization
        if self.use_diff_evolution:
            if self.verbose:
                print(f"  Finding optimal parameters using differential evolution...")
            best_xi, best_tau, best_threshold, best_performance = betasort_analysis.diff_evolution(all_data_df, rat)
        else:
            best_xi = self.default_params['xi']
            best_tau = self.default_params['tau']
            best_threshold = self.default_params['threshold']
        
        results.update({
            'best_xi': best_xi,
            'best_tau': best_tau,
            'best_threshold': best_threshold
        })
        
        if self.verbose:
            print(f"  Using parameters: xi={best_xi}, tau={best_tau}, threshold={best_threshold}")
        
        # Step 2: Run analysis with chosen parameters
        analysis_results = self._analyze_with_parameters(
            all_data_df, rat, best_xi, best_tau, best_threshold
        )
        
        results.update(analysis_results)
        return results
    
    def _analyze_with_parameters(self, all_data_df, rat, xi, tau, threshold):
        """
        Run analysis with specific parameters
        """
        # Initialize storage
        all_models = {}
        global_U, global_L, global_R, global_N = {}, {}, {}, {}
        pair_vte_data = []
        session_results_binomial = {}
        session_results_regression = []
        all_match_rates = []
        adjacent_pair_analysis = {}
        
        # Process each day
        for day, day_data in all_data_df.groupby('Day'):
            if self.verbose:
                print(f"    Processing day {day}...")
            
            day_results = self._process_single_day(
                day, day_data, rat, xi, tau, threshold,
                global_U, global_L, global_R, global_N
            )
            
            # Store results
            all_models[day] = day_results['model']
            pair_vte_data.extend(day_results['vte_data'])
            session_results_binomial[day] = day_results['binomial_results']
            session_results_regression.append(day_results['regression_accuracy'])
            all_match_rates.append(day_results['match_rate'])
            
            # Store adjacent pair data for this day
            adjacent_pair_analysis[day] = day_results['adjacent_pair_data']
            
            # Update global states
            global_U.update(day_results['global_U'])
            global_L.update(day_results['global_L'])
            global_R.update(day_results['global_R'])
            global_N.update(day_results['global_N'])
        
        # Create results dictionary
        pair_vte_df = pd.DataFrame(pair_vte_data)
        if len(pair_vte_df) > 0:
            pair_vte_df['pair'] = pair_vte_df.apply(
                lambda row: f"{row['stim1']}-{row['stim2']}", axis=1
            )
        
        return {
            'all_models': all_models,
            'pair_vte_df': pair_vte_df,
            'session_results_binomial': session_results_binomial,
            'session_predictions_regression': session_results_regression,
            'best_performance': np.mean(all_match_rates),
            'adjacent_pair_analysis': adjacent_pair_analysis
        }
    
    def _process_single_day(self, day, day_data, rat, xi, tau, threshold, 
                           global_U, global_L, global_R, global_N):
        """
        Process a single day of data
        """
        # Extract data
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        rewards = day_data["correct"].values
        
        # Handle VTE data
        if 'VTE' in day_data.columns:
            vtes = day_data["VTE"].values
        else:
            if self.verbose:
                print(f"    WARNING: No VTE column found for {rat} day {day}, defaulting to no VTEs")
            vtes = np.zeros_like(chosen_idx)
            
        # Handle ID data
        if 'ID' in day_data.columns:
            traj_nums = day_data["ID"].values
        else:
            traj_nums = np.arange(len(chosen_idx))
        
        # Initialize model
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        if self.model_type == 'betasort_test':
            model = betasort_test.Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        elif self.model_type == 'betasort_OG':
            model = betasort_OG.Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        else:  # default to betasort
            model = betasort.Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        # Transfer previous state
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
        
        # Reset histories
        model.uncertainty_history = [model.get_all_stimulus_uncertainties()]
        model.ROC_uncertainty_history = [model.get_all_ROC_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.U_history = [model.U.copy()]
        model.L_history = [model.L.copy()]
        
        # Process trials
        day_matches = []
        model_correct_rates = []
        rat_correct_rates = []
        vte_data = []
        
        # Adjacent pair analysis setup
        adjacent_pairs = []
        for i in range(n_stimuli - 1):
            if i in present_stimuli and (i + 1) in present_stimuli:
                adjacent_pairs.append((i, i + 1))
        
        # Pre-update model predictions for adjacent pairs
        pre_update_model_data = {}
        for pair in adjacent_pairs:
            stim1, stim2 = pair
            model_choices = []
            for sim in range(self.n_simulations):
                model_choice = self._model_choose(model, (stim1, stim2), vte=False)
                model_choices.append(model_choice)
            
            correct_rate = np.mean([1 if choice == stim1 else 0 for choice in model_choices])
            pre_update_model_data[pair] = {
                'model_correct_rate': correct_rate,
                'pair_name': f'{stim1}-{stim2}'
            }
        
        # Store actual rat performance for adjacent pairs
        actual_rat_data = {pair: {'rewards': [], 'choices': []} for pair in adjacent_pairs}
        
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            vte = vtes[t]
            traj_num = traj_nums[t]
            
            # Store rat performance for adjacent pairs
            current_pair = (min(chosen, unchosen), max(chosen, unchosen))
            if current_pair in adjacent_pairs:
                actual_rat_data[current_pair]['rewards'].append(reward)
                actual_rat_data[current_pair]['choices'].append(chosen)
            
            # VTE Analysis
            stim1_uncertainty = model.get_uncertainty_stimulus(min(chosen, unchosen))
            stim2_uncertainty = model.get_uncertainty_stimulus(max(chosen, unchosen))
            pair_roc_uncertainty = model.get_uncertainty_ROC(min(chosen, unchosen), max(chosen, unchosen))
            
            vte_data.append({
                'day': day,
                'trial_num': traj_num,
                'stim1': min(chosen, unchosen),
                'stim2': max(chosen, unchosen),
                'chosen': chosen,
                'unchosen': unchosen,
                'vte_occurred': 1 if vte else 0,
                'stim1_uncertainty': stim1_uncertainty,
                'stim2_uncertainty': stim2_uncertainty,
                'pair_roc_uncertainty': pair_roc_uncertainty,
                'reward': reward
            })
            
            # Model simulation
            model_choices = np.zeros(self.n_simulations)
            model_correct = np.zeros(self.n_simulations)
            
            for sim in range(self.n_simulations):
                model_choice = self._model_choose(model, (chosen, unchosen), vte=vte)
                model_choices[sim] = model_choice
                model_correct[sim] = 1 if model_choice == min(chosen, unchosen) else 0
            
            # Calculate metrics
            model_match_rate = np.mean(model_choices == chosen)
            day_matches.append(model_match_rate)
            model_correct_rates.append(np.mean(model_correct))
            rat_correct_rates.append(1 if chosen < unchosen else 0)
            
            # Update model
            model.update(chosen, unchosen, reward, model_match_rate, threshold=threshold)
        
        # Post-update model predictions for adjacent pairs
        post_update_model_data = {}
        for pair in adjacent_pairs:
            stim1, stim2 = pair
            model_choices = []
            for sim in range(self.n_simulations):
                model_choice = self._model_choose(model, (stim1, stim2), vte=False)
                model_choices.append(model_choice)
            
            correct_rate = np.mean([1 if choice == stim1 else 0 for choice in model_choices])
            post_update_model_data[pair] = {
                'model_correct_rate': correct_rate,
                'pair_name': f"{stim1}-{stim2}"
            }
        
        # Calculate actual rat performance for adjacent pairs
        actual_performance = {}
        for pair in adjacent_pairs:
            if len(actual_rat_data[pair]['rewards']) > 0:
                rat_correct_rate = np.mean(actual_rat_data[pair]['rewards'])
                actual_performance[pair] = {
                    'rat_correct_rate': rat_correct_rate,
                    'n_trials': len(actual_rat_data[pair]['rewards']),
                    'pair_name': f"{pair[0]}-{pair[1]}"
                }
            else:
                actual_performance[pair] = {
                    'rat_correct_rate': 0,
                    'n_trials': 0,
                    'pair_name': f"{pair[0]}-{pair[1]}"
                }
        
        # Store adjacent pair analysis data
        adjacent_pair_data = {
            'day': day,
            'adjacent_pairs': adjacent_pairs,
            'pre_update_model': pre_update_model_data,
            'post_update_model': post_update_model_data,
            'actual_rat_performance': actual_performance
        }
        
        # Statistical analysis
        n_rat_correct = int(np.sum(rat_correct_rates))
        n_trials = len(rat_correct_rates)
        model_correct_rate = sum(model_correct_rates) / len(model_correct_rates)
        p_value_binomial = stats.binomtest(n_rat_correct, n_trials, p=model_correct_rate)
        
        binomial_results = {
            'matches': n_rat_correct,
            'trials': n_trials,
            'match_rate': n_rat_correct/n_trials if n_trials > 0 else 0,
            'p_value': p_value_binomial,
            'model_rate': model_correct_rate,
            'significant': p_value_binomial.pvalue < 0.05
        }
        
        # Logistic regression
        X = np.array(model_correct_rates).reshape(-1, 1)
        Y = np.array(rat_correct_rates)
        unique_classes = np.unique(Y)
        
        if len(unique_classes) > 1:
            regression_model = LogisticRegression()
            regression_model.fit(X, Y)
            predictions = regression_model.predict(X)
            accuracy = accuracy_score(Y, predictions)
        else:
            accuracy = float(unique_classes[0])
        
        # Update global states
        new_global_U = {i: model.U[i] for i in range(n_stimuli)}
        new_global_L = {i: model.L[i] for i in range(n_stimuli)}
        new_global_R = {i: model.R[i] for i in range(n_stimuli)}
        new_global_N = {i: model.N[i] for i in range(n_stimuli)}
        
        return {
            'model': model,
            'vte_data': vte_data,
            'binomial_results': binomial_results,
            'regression_accuracy': accuracy,
            'match_rate': np.mean(day_matches),
            'adjacent_pair_data': adjacent_pair_data,
            'global_U': new_global_U,
            'global_L': new_global_L,
            'global_R': new_global_R,
            'global_N': new_global_N
        }
    
    def _save_rat_results(self, rat_name, results):
        """
        Save individual rat results to files
        """
        # Create rat directory
        rat_dir = os.path.join(self.save_path, rat_name)
        os.makedirs(rat_dir, exist_ok=True)
        
        # Check transitive inference
        all_models = results["all_models"]
        final_day = max(all_models.keys())
        ti_result = betasort_analysis.check_transitive_inference(
            all_models[final_day], test=(self.model_type in ['betasort_test', 'betasort_OG'])
        )
        ti_result_serializable = {f"{k[0]},{k[1]}": v for k, v in ti_result.items()}
        
        # Check real transitive inference if data exists
        real_ti_data_path = os.path.join(self.data_path, "inferenceTesting", rat_name, f"{rat_name}.csv")
        if os.path.exists(real_ti_data_path):
            real_ti_data = pd.read_csv(real_ti_data_path)
            ti_result_real = betasort_analysis.check_transitive_inference_real(
                all_models[final_day], real_ti_data, test=(self.model_type in ['betasort_test', 'betasort_OG'])
            )
        else:
            if self.verbose:
                print(f"    No inference testing data found for {rat_name}")
            ti_result_real = {}
        
        # Analyze correlations
        pair_results = betasort_analysis.analyze_correlations(results["pair_vte_df"])
        
        # Prepare summary results
        summary_results = {
            "rat": rat_name,
            "model_type": self.model_type,  # Store which model was used
            "best_xi": results["best_xi"],
            "best_tau": results["best_tau"],
            "best_threshold": results["best_threshold"],
            "best_performance": results["best_performance"],
            "session_regression": results["session_predictions_regression"],
            "session_binomial_test": results["session_results_binomial"],
            "TI_Result": json.dumps(ti_result_serializable)
        }
        
        # Save files
        results_df = pd.DataFrame([summary_results])
        results_filename = f"{self.model_type}_results.csv"
        results_df.to_csv(os.path.join(rat_dir, results_filename), index=False)
        
        results["pair_vte_df"].to_csv(os.path.join(rat_dir, "vte_uncertainty.csv"), index=False)
        
        with open(os.path.join(rat_dir, "uncertainty_vte.json"), 'w') as f:
            json.dump(pair_results, f, indent=2)
        
        # Save adjacent pair analysis data
        adjacent_pair_filename = f"{self.model_type}_adjacent_pair_analysis.json"
        with open(os.path.join(rat_dir, adjacent_pair_filename), 'w') as f:
            # Convert adjacent_pair_analysis to JSON-serializable format
            serializable_adj_data = {}
            for day, day_data in results['adjacent_pair_analysis'].items():
                serializable_adj_data[str(day)] = {
                    'day': day_data['day'],
                    'adjacent_pairs': [list(pair) for pair in day_data['adjacent_pairs']],  # Convert tuples to lists
                    'pre_update_model': {f"{k[0]}-{k[1]}": v for k, v in day_data['pre_update_model'].items()},
                    'post_update_model': {f"{k[0]}-{k[1]}": v for k, v in day_data['post_update_model'].items()},
                    'actual_rat_performance': {f"{k[0]}-{k[1]}": v for k, v in day_data['actual_rat_performance'].items()}
                }
            json.dump(serializable_adj_data, f, indent=2)
        
        # Save transitive inference results
        ti_filename = f"{self.model_type}_ti_results.json"
        with open(os.path.join(rat_dir, ti_filename), 'w') as f:
            # Convert ti_result_real to JSON-serializable format
            serializable_ti_data = {}
            for pair, (model_pct, rat_pct, n) in ti_result_real.items():
                pair_key = f"{pair[0]}-{pair[1]}" if isinstance(pair, tuple) else str(pair)
                serializable_ti_data[pair_key] = {
                    'model_pct': model_pct,
                    'rat_pct': rat_pct,
                    'n_trials': n
                }
            json.dump(serializable_ti_data, f, indent=2)
        
        # Store for aggregation
        self.rat_results[rat_name] = {
            'results': results,
            'ti_result_real': ti_result_real,
            'summary': summary_results
        }
        
        if self.verbose:
            print(f"  Saved results for {rat_name} (performance: {results['best_performance']:.3f})")
    
    def aggregate_results(self, rats_to_include=None, rats_to_exclude=None):
        """
        Aggregate results across selected rats
        
        Parameters:
        -----------
        rats_to_include : list, optional
            Specific rats to include in aggregation. If None, uses all available rats.
        rats_to_exclude : list, optional
            Specific rats to exclude from aggregation.
        """
        # Filter rats for aggregation
        available_rats = list(self.rat_results.keys())
        rats_for_aggregation = self._filter_rats_for_aggregation(
            available_rats, rats_to_include, rats_to_exclude
        )
        
        if self.verbose:
            print(f"Aggregating results for {len(rats_for_aggregation)} rats: {rats_for_aggregation}")
            if rats_to_exclude:
                excluded = [r for r in available_rats if r not in rats_for_aggregation]
                print(f"Excluding: {excluded}")
        
        # Create filtered rat_results for aggregation
        filtered_rat_results = {rat: self.rat_results[rat] for rat in rats_for_aggregation}
        
        # Aggregate adjacent pair analysis (only if data was analyzed, not loaded)
        if any('results' in rat_data for rat_data in filtered_rat_results.values()):
            self._aggregate_adjacent_pair_analysis(filtered_rat_results)
        elif self.verbose:
            print("  Skipping adjacent pair analysis - not available in loaded data")
        
        # Aggregate transitive inference results
        self._aggregate_transitive_inference(filtered_rat_results)
        
        if self.verbose:
            print(f"Aggregated results for {len(rats_for_aggregation)} rats")
    
    def _filter_rats_for_aggregation(self, available_rats, rats_to_include, rats_to_exclude):
        """
        Filter rats based on include/exclude criteria
        """
        # Apply inclusion filter
        if rats_to_include is not None:
            rats = [rat for rat in available_rats if rat in rats_to_include]
        else:
            rats = available_rats.copy()
        
        # Apply exclusion filter
        if rats_to_exclude is not None:
            rats = [rat for rat in rats if rat not in rats_to_exclude]
        
        return rats
    
    def _aggregate_adjacent_pair_analysis(self, filtered_rat_results):
        """
        Aggregate adjacent pair analysis across rats
        """
        if self.verbose:
            print("  Aggregating adjacent pair analysis...")
        
        # Collect adjacent pair data from filtered rats (using final day for each rat)
        all_rats_adjacent_data = {}
        
        for rat_name, rat_data in filtered_rat_results.items():
            results = rat_data['results']
            if 'adjacent_pair_analysis' in results:
                # Get the last day for this rat
                last_day = max(results['adjacent_pair_analysis'].keys())
                last_day_data = results['adjacent_pair_analysis'][last_day]
                
                all_rats_adjacent_data[rat_name] = {
                    'last_day': last_day,
                    'data': last_day_data
                }
        
        # Find all unique pairs across all rats
        all_pairs = set()
        for rat_data in all_rats_adjacent_data.values():
            all_pairs.update(rat_data['data']['adjacent_pairs'])
        
        # Convert to sorted list for consistent ordering
        all_pairs = sorted(list(all_pairs))
        pair_names = [f"{p[0]}-{p[1]}" for p in all_pairs]
        
        if self.verbose:
            print(f"    Found pairs across all rats: {pair_names}")
        
        # Initialize storage for averaged data
        aggregated_data = {
            'pair_names': pair_names,
            'rat_rates': [],
            'pre_model_rates': [],
            'post_model_rates': [],
            'rat_counts': [],  # Number of rats that had data for each pair
        }
        
        # Calculate averages for each pair
        for pair in all_pairs:
            rat_rates_for_pair = []
            pre_model_rates_for_pair = []
            post_model_rates_for_pair = []
            
            for rat, rat_info in all_rats_adjacent_data.items():
                rat_data = rat_info['data']
                
                if pair in rat_data['adjacent_pairs']:
                    # Get rat performance
                    if pair in rat_data['actual_rat_performance']:
                        rat_rates_for_pair.append(rat_data['actual_rat_performance'][pair]['rat_correct_rate'])
                    
                    # Get pre-update model performance
                    if pair in rat_data['pre_update_model']:
                        pre_model_rates_for_pair.append(rat_data['pre_update_model'][pair]['model_correct_rate'])
                    
                    # Get post-update model performance
                    if pair in rat_data['post_update_model']:
                        post_model_rates_for_pair.append(rat_data['post_update_model'][pair]['model_correct_rate'])
            
            # Calculate averages
            aggregated_data['rat_rates'].append(np.mean(rat_rates_for_pair) if rat_rates_for_pair else 0)
            aggregated_data['pre_model_rates'].append(np.mean(pre_model_rates_for_pair) if pre_model_rates_for_pair else 0)
            aggregated_data['post_model_rates'].append(np.mean(post_model_rates_for_pair) if post_model_rates_for_pair else 0)
            aggregated_data['rat_counts'].append(len(rat_rates_for_pair))
            
            if self.verbose:
                print(f"    Pair {pair[0]}-{pair[1]}: {len(rat_rates_for_pair)} rats, "
                      f"Rat avg: {np.mean(rat_rates_for_pair):.3f}, "
                      f"Pre-model avg: {np.mean(pre_model_rates_for_pair):.3f}, "
                      f"Post-model avg: {np.mean(post_model_rates_for_pair):.3f}")
        
        # Save aggregated data
        aggregated_results_path = os.path.join(self.save_path, "aggregated_adjacent_pair_analysis.json")
        with open(aggregated_results_path, 'w') as f:
            json.dump({
                'pair_names': aggregated_data['pair_names'],
                'rat_rates': aggregated_data['rat_rates'],
                'pre_model_rates': aggregated_data['pre_model_rates'],
                'post_model_rates': aggregated_data['post_model_rates'],
                'rat_counts': aggregated_data['rat_counts'],
                'rats_included': list(all_rats_adjacent_data.keys()),
                'total_rats': len(all_rats_adjacent_data)
            }, f, indent=2)
        
        # Store individual data for box plots
        individual_rat_data = {}
        individual_model_data = {}
        
        for pair in all_pairs:
            pair_name = f"{pair[0]}-{pair[1]}"
            individual_rat_data[pair_name] = []
            individual_model_data[pair_name] = []
            
            for rat, rat_info in all_rats_adjacent_data.items():
                rat_data = rat_info['data']
                
                if pair in rat_data['adjacent_pairs']:
                    # Get rat performance
                    if pair in rat_data['actual_rat_performance']:
                        individual_rat_data[pair_name].append(rat_data['actual_rat_performance'][pair]['rat_correct_rate'])
                    
                    # Get post-update model performance (using post-update as the primary comparison)
                    if pair in rat_data['post_update_model']:
                        individual_model_data[pair_name].append(rat_data['post_update_model'][pair]['model_correct_rate'])
        
        # Store for plotting
        self.aggregated_data['adjacent_pair_results'] = aggregated_data
        self.aggregated_data['adjacent_pair_results']['total_rats'] = len(filtered_rat_results)
        self.aggregated_data['adjacent_pair_results']['individual_rat_data'] = individual_rat_data
        self.aggregated_data['adjacent_pair_results']['individual_model_data'] = individual_model_data
    
    def _model_choose(self, model, stimuli, vte=False):
        """
        Helper function to call model.choose() with appropriate parameters for each model type
        
        Parameters:
        -----------
        model : Betasort model instance
        stimuli : tuple or list
            Either (stim1, stim2) for betasort_test/betasort_OG or [stim1, stim2] for betasort
        vte : bool
            VTE flag (only used by betasort_test and betasort_OG)
        
        Returns:
        --------
        int : Chosen stimulus index
        """
        # These models take (chosen, unchosen, vte) parameters
        if isinstance(stimuli, (list, tuple)) and len(stimuli) == 2:
            return model.choose(stimuli[0], stimuli[1], vte)
        else:
            raise ValueError(f"Expected 2 stimuli for {self.model_type}, got {stimuli}")
    
    def _aggregate_transitive_inference(self, filtered_rat_results):
        """
        Aggregate transitive inference results across rats
        """
        if self.verbose:
            print("  Aggregating transitive inference results...")
        
        # Collect TI results from filtered rats
        type_to_model = defaultdict(list)
        type_to_rat = defaultdict(list)
        type_to_n = defaultdict(list)
        
        for rat_name, rat_data in filtered_rat_results.items():
            ti_dict = rat_data.get('ti_result_real', {})
            for pair, (model_pct, rat_pct, n) in ti_dict.items():
                type_to_model[pair].append(model_pct)
                type_to_rat[pair].append(rat_pct)
                type_to_n[pair].append(n)
        
        # Prepare aggregated data
        trial_types = sorted(type_to_model.keys())
        model_means = [np.mean(type_to_model[pair]) for pair in trial_types]
        rat_means = [np.mean(type_to_rat[pair]) for pair in trial_types]
        model_sems = [np.std(type_to_model[pair], ddof=1)/np.sqrt(len(type_to_model[pair])) 
                     for pair in trial_types]
        rat_sems = [np.std(type_to_rat[pair], ddof=1)/np.sqrt(len(type_to_rat[pair])) 
                   for pair in trial_types]
        labels = [f"{pair[0]}-{pair[1]}" for pair in trial_types]
        
        # Save aggregated data
        agg_df = pd.DataFrame({
            'trial_type': labels,
            'model_mean': model_means,
            'model_sem': model_sems,
            'rat_mean': rat_means,
            'rat_sem': rat_sems,
            'n_rats': [len(type_to_model[pair]) for pair in trial_types]
        })
        
        agg_csv_path = os.path.join(self.save_path, 'aggregated_ti_real_results.csv')
        agg_df.to_csv(agg_csv_path, index=False)
        
        # Store for plotting
        self.aggregated_data['ti_results'] = {
            'trial_types': trial_types,
            'model_means': model_means,
            'rat_means': rat_means,
            'model_sems': model_sems,
            'rat_sems': rat_sems,
            'labels': labels,
            'individual_rat_data': type_to_rat,
            'individual_model_data': type_to_model
        }
    
    def generate_plots(self):
        """
        Generate all plots and figures
        """
        if self.verbose:
            print("Generating plots...")
        
        # Create plots directory
        plots_dir = os.path.join(self.save_path, "aggregated_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate transitive inference plot
        self._plot_transitive_inference(plots_dir)
        
        # Generate adjacent pair plots
        self._plot_adjacent_pair_analysis(plots_dir)
        
        if self.verbose:
            print(f"Saved plots to {plots_dir}")
    
    def _plot_transitive_inference(self, plots_dir):
        """
        Generate transitive inference comparison plot with box plots and overlaid model results
        """
        if 'ti_results' not in self.aggregated_data:
            return
        
        ti_data = self.aggregated_data['ti_results']
        
        # Check if we have individual data for box plots
        if 'individual_rat_data' not in ti_data or 'individual_model_data' not in ti_data:
            # Fallback to original bar plot if individual data not available
            self._plot_transitive_inference_original(plots_dir)
            return
            
        # Create box plot with overlaid model results
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define desired pairs in order: AB, BC, CD, DE, BD, AE
        desired_pairs = ['AB', 'BC', 'CD', 'DE', 'BD', 'AE']
        
        # Convert pairs to trial type tuples for lookup
        rat_data_for_plot = []
        model_data_for_plot = []
        labels = []
        
        for pair in desired_pairs:
            if len(pair) == 2:
                # Convert letter pair to numbers
                num1 = HIERARCHY_MAPPINGS[pair[0]]
                num2 = HIERARCHY_MAPPINGS[pair[1]]
                
                # Try both orderings as keys
                trial_key1 = (num1, num2)
                trial_key2 = (num2, num1)
                
                trial_key = None
                if trial_key1 in ti_data['individual_rat_data']:
                    trial_key = trial_key1
                elif trial_key2 in ti_data['individual_rat_data']:
                    trial_key = trial_key2
                
                if (trial_key and 
                    trial_key in ti_data['individual_rat_data'] and 
                    trial_key in ti_data['individual_model_data'] and
                    len(ti_data['individual_rat_data'][trial_key]) > 0 and
                    len(ti_data['individual_model_data'][trial_key]) > 0):
                    
                    rat_data_for_plot.append(ti_data['individual_rat_data'][trial_key])
                    model_data_for_plot.append(ti_data['individual_model_data'][trial_key])
                    labels.append(pair)
        
        if not rat_data_for_plot:
            if self.verbose:
                print("  No individual data available for TI box plots")
            return
        
        x_positions = np.arange(len(labels))
        
        # Create box plots for rat data (wider, behind)
        bp_rats = ax.boxplot(rat_data_for_plot, positions=x_positions, widths=0.6, 
                            patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='blue', linewidth=2),
                            whiskerprops=dict(visible=False),  # Remove whiskers
                            capprops=dict(visible=False),      # Remove caps
                            flierprops=dict(visible=False))    # Remove outliers
        
        # Create box plots for model data (narrower, overlaid on top)
        bp_models = ax.boxplot(model_data_for_plot, positions=x_positions, widths=0.3, 
                              patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.8),
                              medianprops=dict(color='darkgreen', linewidth=2),
                              whiskerprops=dict(visible=False),  # Remove whiskers
                              capprops=dict(visible=False),      # Remove caps
                              flierprops=dict(visible=False))    # Remove outliers
        
        # Create positions with spacing: AB BC CD DE (gap) BD (gap) AE
        x_positions = []
        current_pos = 0
        
        for i, label in enumerate(labels):
            x_positions.append(current_pos)
            current_pos += 1
            
            # Add extra space after DE and BD
            if label == 'DE' or label == 'BD':
                current_pos += 0.5
        
        x_positions = np.array(x_positions)
        
        # Clear and recreate plots with proper positions
        ax.clear()
        
        # Create box plots with custom spacing
        bp_rats = ax.boxplot(rat_data_for_plot, positions=x_positions, widths=0.6, 
                            patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='blue', linewidth=2),
                            whiskerprops=dict(visible=False),
                            capprops=dict(visible=False),
                            flierprops=dict(visible=False))
        
        bp_models = ax.boxplot(model_data_for_plot, positions=x_positions, widths=0.3, 
                              patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.8),
                              medianprops=dict(color='darkgreen', linewidth=2),
                              whiskerprops=dict(visible=False),
                              capprops=dict(visible=False),
                              flierprops=dict(visible=False))
        
        ax.set_xlabel('Trial Type', fontsize=22)
        ax.set_ylabel('Percent Correct', fontsize=22)
        ax.set_title('Transitive Inference: Rat Data vs Model Results', fontsize=24)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=16)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='Rat Data'),
                          Patch(facecolor='lightgreen', alpha=0.8, label='Model Results')]
        ax.legend(handles=legend_elements, fontsize=16)
        ax.set_ylim(0, 1.05)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='Rat Data'),
                          Patch(facecolor='lightgreen', alpha=0.8, label='Model Results')]
        ax.legend(handles=legend_elements, fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Update legend font size
        ax.legend(handles=legend_elements, fontsize=16)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'aggregated_ti_real_boxplot.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        if self.verbose:
            print(f"  Saved TI box plot: {plot_path}")
    
    def _plot_transitive_inference_original(self, plots_dir):
        """
        Generate original transitive inference comparison plot (fallback)
        """
        ti_data = self.aggregated_data['ti_results']
        
        # Create plot
        x = np.arange(len(ti_data['labels']))
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 7))
        
        rects1 = ax.bar(x - width/2, ti_data['rat_means'], width, 
                       yerr=ti_data['rat_sems'], label='Rat', 
                       color='blue', alpha=0.7, capsize=5)
        rects2 = ax.bar(x + width/2, ti_data['model_means'], width, 
                       yerr=ti_data['model_sems'], label='Model', 
                       color='green', alpha=0.7, capsize=5)
        
        ax.set_xlabel('Trial Type', fontsize=14)
        ax.set_ylabel('Percent Correct', fontsize=14)
        ax.set_title('Transitive Inference: Model vs Rat (Averaged Across Rats)', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(ti_data['labels'], fontsize=12)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.05)
        
        # Add value labels
        for rects, means in zip([rects1, rects2], [ti_data['rat_means'], ti_data['model_means']]):
            for rect, mean in zip(rects, means):
                height = rect.get_height()
                ax.annotate(f'{mean:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'aggregated_ti_real_plot.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        if self.verbose:
            print(f"  Saved TI plot: {plot_path}")
    
    def _plot_adjacent_pair_analysis(self, plots_dir):
        """
        Generate adjacent pair comparison plots with box plots and overlaid model results
        """
        if 'adjacent_pair_results' not in self.aggregated_data:
            return
        
        adj_data = self.aggregated_data['adjacent_pair_results']
        
        # Create new box plot with overlaid model results
        if 'individual_rat_data' in adj_data and 'individual_model_data' in adj_data:
            try:
                self._plot_adjacent_pair_boxplot(plots_dir, adj_data)
            except Exception as e:
                print(f"  Error creating adjacent pair box plot: {e}")
        
        # Create the aggregated comparison plot (3-way comparison)
        try:
            betasort_plots.plot_aggregated_adjacent_pair_comparison(
                adj_data['pair_names'],
                adj_data['rat_rates'],
                adj_data['pre_model_rates'],
                adj_data['post_model_rates'],
                adj_data['rat_counts'],
                total_rats=self.aggregated_data['adjacent_pair_results']['total_rats'],
                save=os.path.join(plots_dir, "all_rats_adjacent_pair_comparison_plot_aggregated_adjacent_pair_comparison.png")
            )
            
            if self.verbose:
                print(f"  Saved aggregated adjacent pair comparison plot")
        except Exception as e:
            print(f"  Error creating aggregated adjacent pair plot: {e}")
        
        # Create the simplified comparison plot (post-model vs rat only)
        try:
            betasort_plots.plot_post_model_vs_rat_comparison(
                adj_data['pair_names'],
                adj_data['rat_rates'],
                adj_data['post_model_rates'],
                adj_data['rat_counts'],
                total_rats=self.aggregated_data['adjacent_pair_results']['total_rats'],
                save=os.path.join(plots_dir, "all_rats_post_model_vs_rat_comparison_plot_post_model_vs_rat_comparison.png")
            )
            
            if self.verbose:
                print(f"  Saved post-model vs rat comparison plot")
        except Exception as e:
            print(f"  Error creating post-model vs rat plot: {e}")
    
    def _plot_adjacent_pair_boxplot(self, plots_dir, adj_data):
        """
        Create box plot for adjacent pair analysis with overlaid model results
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define desired pairs in order: AB, BC, CD, DE, BD, AE
        desired_pairs = ['AB', 'BC', 'CD', 'DE', 'BD', 'AE']
        
        # Filter and reorder data based on desired pairs
        rat_data_for_plot = []
        model_data_for_plot = []
        valid_pair_names = []
        
        for pair in desired_pairs:
            if len(pair) == 2:
                # Convert letter pair to numeric format for lookup
                num1 = HIERARCHY_MAPPINGS[pair[0]]
                num2 = HIERARCHY_MAPPINGS[pair[1]]
                numeric_key = f"{num1}-{num2}"
                
                if (numeric_key in adj_data['individual_rat_data'] and 
                    numeric_key in adj_data['individual_model_data'] and
                    len(adj_data['individual_rat_data'][numeric_key]) > 0 and
                    len(adj_data['individual_model_data'][numeric_key]) > 0):
                    
                    rat_data_for_plot.append(adj_data['individual_rat_data'][numeric_key])
                    model_data_for_plot.append(adj_data['individual_model_data'][numeric_key])
                    valid_pair_names.append(pair)
        
        if not rat_data_for_plot:
            if self.verbose:
                print("  No individual data available for adjacent pair box plots")
            return
        
        x_positions = np.arange(len(valid_pair_names))
        
        # Create box plots for rat data (wider, behind)
        bp_rats = ax.boxplot(rat_data_for_plot, positions=x_positions, widths=0.6, 
                            patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='blue', linewidth=2),
                            whiskerprops=dict(visible=False),  # Remove whiskers
                            capprops=dict(visible=False),      # Remove caps
                            flierprops=dict(visible=False))    # Remove outliers
        
        # Create box plots for model data (narrower, overlaid on top)
        bp_models = ax.boxplot(model_data_for_plot, positions=x_positions, widths=0.3, 
                              patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.8),
                              medianprops=dict(color='darkgreen', linewidth=2),
                              whiskerprops=dict(visible=False),  # Remove whiskers
                              capprops=dict(visible=False),      # Remove caps
                              flierprops=dict(visible=False))    # Remove outliers
        
        # Create positions with spacing: AB BC CD DE (gap) BD (gap) AE
        x_positions = []
        current_pos = 0
        
        for i, label in enumerate(valid_pair_names):
            x_positions.append(current_pos)
            current_pos += 1
            
            # Add extra space after DE and BD
            if label == 'DE' or label == 'BD':
                current_pos += 0.5
        
        x_positions = np.array(x_positions)
        
        # Clear and recreate plots with proper positions
        ax.clear()
        
        # Create box plots with custom spacing
        bp_rats = ax.boxplot(rat_data_for_plot, positions=x_positions, widths=0.6, 
                            patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='blue', linewidth=2),
                            whiskerprops=dict(visible=False),
                            capprops=dict(visible=False),
                            flierprops=dict(visible=False))
        
        bp_models = ax.boxplot(model_data_for_plot, positions=x_positions, widths=0.3, 
                              patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.8),
                              medianprops=dict(color='darkgreen', linewidth=2),
                              whiskerprops=dict(visible=False),
                              capprops=dict(visible=False),
                              flierprops=dict(visible=False))
        
        ax.set_xlabel('Adjacent Stimulus Pairs', fontsize=22)
        ax.set_ylabel('Correct Choice Rate', fontsize=22)
        ax.set_title('Adjacent Pair Performance: Rat Data vs Model Results', fontsize=24)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(valid_pair_names, fontsize=16)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='Rat Data'),
                          Patch(facecolor='lightgreen', alpha=0.8, label='Model Results')]
        ax.legend(handles=legend_elements, fontsize=16)
        
        # Make y-tick labels larger
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylim(0, 1.05)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='Rat Data'),
                          Patch(facecolor='lightgreen', alpha=0.8, label='Model Results')]
        ax.legend(handles=legend_elements, fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Update legend font size
        ax.legend(handles=legend_elements, fontsize=16)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'adjacent_pair_boxplot_with_model_overlay.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        if self.verbose:
            print(f"  Saved adjacent pair box plot: {plot_path}")
    
    def run_analysis_only(self, rats_to_include=None, rats_to_exclude=None):
        """
        Run only the analysis phase (no aggregation or plotting)
        
        Parameters:
        -----------
        rats_to_include : list, optional
            Specific rats to include in analysis. If None, includes all rats.
        rats_to_exclude : list, optional
            Specific rats to exclude from analysis.
        """
        if self.verbose:
            print("=" * 60)
            print("RUNNING BETASORT ANALYSIS (ANALYSIS ONLY)")
            print("=" * 60)
        
        # Temporarily store original filters and set new ones
        original_include = self.rats_to_include
        original_exclude = self.rats_to_exclude
        
        self.rats_to_include = rats_to_include if rats_to_include is not None else self.rats_to_include
        self.rats_to_exclude = rats_to_exclude if rats_to_exclude is not None else self.rats_to_exclude
        
        # Get list of rats to process
        rats_to_process = self.get_rats_to_process()
        
        if not rats_to_process:
            print("No rats to process!")
            return
        
        # Process each rat
        for rat_name in rats_to_process:
            # Find rat data file
            rat_path = os.path.join(self.data_path, rat_name)
            rat_file = None
            
            for root, _, files in os.walk(rat_path):
                for file in files:
                    if (".DS_Store" not in file and 
                        "zIdPhi" not in file and 
                        "all_days" in file and 
                        file.endswith('.csv')):
                        rat_file = os.path.join(root, file)
                        break
                if rat_file:
                    break
            
            if not rat_file:
                print(f"No data file found for {rat_name}")
                continue
            
            # Load and analyze rat data
            try:
                rat_data = pd.read_csv(rat_file)
                self.analyze_single_rat(rat_name, rat_data)
            except Exception as e:
                print(f"Error processing {rat_name}: {e}")
                continue
        
        # Restore original filters
        self.rats_to_include = original_include
        self.rats_to_exclude = original_exclude
        
        if self.verbose:
            print("=" * 60)
            print("ANALYSIS PHASE COMPLETE")
            print(f"Processed {len(self.rat_results)} rats successfully")
            print("Data saved individually. Use aggregate_and_plot() for aggregation.")
            print("=" * 60)
    
    def aggregate_and_plot(self, rats_to_include=None, rats_to_exclude=None, 
                          output_suffix="", plots_dir=None):
        """
        Aggregate results and generate plots for selected rats
        
        Parameters:
        -----------
        rats_to_include : list, optional
            Specific rats to include in aggregation. If None, uses all analyzed rats.
        rats_to_exclude : list, optional
            Specific rats to exclude from aggregation.
        output_suffix : str, optional
            Suffix to add to output files (e.g., "_no_BP09")
        plots_dir : str, optional
            Custom directory for plots. If None, uses default with suffix.
        """
        if not self.rat_results:
            print("No rat results available. Run analysis first!")
            return
        
        if self.verbose:
            print("=" * 60)
            print("AGGREGATING AND PLOTTING RESULTS")
            print("=" * 60)
        
        # Aggregate results with filtering
        self.aggregate_results(rats_to_include, rats_to_exclude)
        
        # Generate plots in custom directory if specified
        if plots_dir is None:
            plots_dir = os.path.join(self.save_path, f"aggregated_plots{output_suffix}")
        
        # Create plots directory
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate plots
        self._plot_transitive_inference(plots_dir)
        self._plot_adjacent_pair_analysis(plots_dir)
        
        if self.verbose:
            print(f"Plots saved to: {plots_dir}")
            print("=" * 60)
    
    def run_full_analysis(self):
        """
        Run the complete analysis pipeline (for backward compatibility)
        """
        if self.verbose:
            print("=" * 60)
            print("STARTING BETASORT ANALYSIS PIPELINE")
            print("=" * 60)
        
        # Run analysis only
        self.run_analysis_only(self.rats_to_include, self.rats_to_exclude)
        
        # Aggregate and plot with the same filters
        self.aggregate_and_plot(self.rats_to_include, self.rats_to_exclude)
        
        if self.verbose:
            print("=" * 60)
            print("FULL ANALYSIS COMPLETE")
            print(f"Results saved to: {self.save_path}")
            print("=" * 60)
    
    @classmethod
    def from_saved_data(cls, save_path=None, data_path=None, verbose=True):
        """
        Create a pipeline instance from previously saved rat data
        
        Parameters:
        -----------
        save_path : str, optional
            Path where rat results were saved
        data_path : str, optional  
            Path to original data (for potential re-analysis)
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        BetasortPipeline : Pipeline instance with loaded rat results
        """
        if save_path is None:
            save_path = paths.betasort_data
        if data_path is None:
            data_path = paths.preprocessed_data_model
            
        # Create pipeline instance (model_type will be detected from saved data)
        pipeline = cls(
            data_path=data_path,
            save_path=save_path,
            verbose=verbose
        )
        
        # Load existing rat results
        pipeline._load_saved_rat_results()
        
        return pipeline
    
    def _load_saved_rat_results(self):
        """
        Load previously saved rat results from disk
        """
        if self.verbose:
            print("Loading saved rat results...")
        
        rat_dirs = [d for d in os.listdir(self.save_path) 
                   if os.path.isdir(os.path.join(self.save_path, d))
                   and d != "aggregated_plots"]
        
        detected_model_types = set()
        
        for rat_name in rat_dirs:
            rat_dir = os.path.join(self.save_path, rat_name)
            vte_file = os.path.join(rat_dir, "vte_uncertainty.csv")
            uncertainty_file = os.path.join(rat_dir, "uncertainty_vte.json")
            
            # Look for model-specific results files (new format) or fallback to old format
            results_file = None
            adjacent_pair_file = None
            ti_results_file = None
            detected_model_type = None
            
            for model_type in ['betasort_test', 'betasort_OG', 'betasort']:
                potential_file = os.path.join(rat_dir, f"{model_type}_results.csv")
                potential_adj_file = os.path.join(rat_dir, f"{model_type}_adjacent_pair_analysis.json")
                potential_ti_file = os.path.join(rat_dir, f"{model_type}_ti_results.json")
                if os.path.exists(potential_file):
                    results_file = potential_file
                    detected_model_type = model_type
                    if os.path.exists(potential_adj_file):
                        adjacent_pair_file = potential_adj_file
                    if os.path.exists(potential_ti_file):
                        ti_results_file = potential_ti_file
                    break
            
            # Fallback to old filename
            if results_file is None:
                old_results_file = os.path.join(rat_dir, "results.csv")
                if os.path.exists(old_results_file):
                    results_file = old_results_file
            
            if results_file:
                try:
                    # Load summary results
                    summary_df = pd.read_csv(results_file)
                    summary = summary_df.iloc[0].to_dict()
                    
                    # Detect model type from saved data
                    if 'model_type' in summary:
                        detected_model_types.add(summary['model_type'])
                    
                    # Load VTE data
                    vte_df = pd.read_csv(vte_file) if os.path.exists(vte_file) else pd.DataFrame()
                    
                    # Load uncertainty correlation data
                    with open(uncertainty_file, 'r') as f:
                        uncertainty_corr = json.load(f)
                    
                    # Load adjacent pair analysis data if available
                    adjacent_pair_analysis = {}
                    if adjacent_pair_file and os.path.exists(adjacent_pair_file):
                        with open(adjacent_pair_file, 'r') as f:
                            adj_data_json = json.load(f)
                            # Convert back from JSON format to expected structure
                            for day_str, day_data in adj_data_json.items():
                                day = int(day_str)
                                adjacent_pair_analysis[day] = {
                                    'day': day_data['day'],
                                    'adjacent_pairs': [tuple(pair) for pair in day_data['adjacent_pairs']],  # Convert lists back to tuples
                                    'pre_update_model': {tuple(map(int, k.split('-'))): v for k, v in day_data['pre_update_model'].items()},
                                    'post_update_model': {tuple(map(int, k.split('-'))): v for k, v in day_data['post_update_model'].items()},
                                    'actual_rat_performance': {tuple(map(int, k.split('-'))): v for k, v in day_data['actual_rat_performance'].items()}
                                }
                    
                    # Load transitive inference results if available
                    ti_result_real = {}
                    if ti_results_file and os.path.exists(ti_results_file):
                        with open(ti_results_file, 'r') as f:
                            ti_data_json = json.load(f)
                            # Convert back from JSON format to expected structure
                            for pair_key, ti_data in ti_data_json.items():
                                # Convert "0-1" back to (0, 1) tuple
                                pair = tuple(map(int, pair_key.split('-')))
                                ti_result_real[pair] = (
                                    ti_data['model_pct'],
                                    ti_data['rat_pct'],
                                    ti_data['n_trials']
                                )
                    
                    # Store in rat_results (simplified structure for loading)
                    self.rat_results[rat_name] = {
                        'summary': summary,
                        'vte_df': vte_df,
                        'uncertainty_corr': uncertainty_corr,
                        'ti_result_real': ti_result_real,
                        'results': {'adjacent_pair_analysis': adjacent_pair_analysis} if adjacent_pair_analysis else {}
                    }
                    
                    if self.verbose:
                        print(f"  Loaded {rat_name}")
                        
                except Exception as e:
                    print(f"  Error loading {rat_name}: {e}")
        
        # Set model type based on detected types
        if len(detected_model_types) == 1:
            self.model_type = detected_model_types.pop()
            if self.verbose:
                print(f"Detected model type: {self.model_type}")
        elif len(detected_model_types) > 1:
            if self.verbose:
                print(f"WARNING: Multiple model types detected: {detected_model_types}")
                print(f"Using default: {self.model_type}")
        elif len(detected_model_types) == 0:
            if self.verbose:
                print(f"No model type found in saved data, using default: {self.model_type}")
        
        if self.verbose:
            print(f"Loaded {len(self.rat_results)} rats from saved data")


def quick_analysis_all_rats(model_type='betasort_test', verbose=True):
    """
    Quick function to analyze ALL rats (no exclusions)
    
    Parameters:
    -----------
    model_type : str
        Model type to use ('betasort', 'betasort_test', or 'betasort_OG')
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    BetasortPipeline : The pipeline object with results
    """
    pipeline = BetasortPipeline(
        model_type=model_type,
        use_diff_evolution=False,
        verbose=verbose
    )
    
    pipeline.run_analysis_only()
    return pipeline


def quick_analysis(rats_to_exclude=None, model_type='betasort_test', verbose=True):
    """
    Quick function to run full analysis with common settings (backward compatibility)
    
    Parameters:
    -----------
    rats_to_exclude : list, optional
        Rats to exclude (default: ['inferenceTesting'])
    model_type : str
        Model type to use ('betasort', 'betasort_test', or 'betasort_OG')
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    BetasortPipeline : The pipeline object with results
    """
    if rats_to_exclude is None:
        rats_to_exclude = ['inferenceTesting']
    
    pipeline = BetasortPipeline(
        rats_to_exclude=rats_to_exclude,
        model_type=model_type,
        use_diff_evolution=False,
        verbose=verbose
    )
    
    pipeline.run_full_analysis()
    return pipeline


def analyze_example_rats(model_type='betasort_test', n_rats=10, use_diff_evolution=False, verbose=True):
    """
    Helper function to easily analyze all example rats.
    
    Parameters:
    -----------
    model_type : str
        Model type to use ('betasort', 'betasort_test', or 'betasort_OG')
    n_rats : int
        Number of example rats to analyze (default 10)
    use_diff_evolution : bool
        Whether to use differential evolution for parameter optimization
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    BetasortPipeline : The pipeline object with results
    """
    # Generate list of example rat names
    example_rats = [f"ExampleRat{i}" for i in range(1, n_rats + 1)]
    
    print(f"Analyzing {n_rats} example rats with {model_type} model...")
    print(f"Example rats: {example_rats}")
    
    pipeline = BetasortPipeline(
        rats_to_include=example_rats,
        rats_to_exclude=['inferenceTesting'],  # Exclude the inference testing directory
        model_type=model_type,
        use_diff_evolution=use_diff_evolution,
        verbose=verbose
    )
    
    # Run full analysis
    pipeline.run_full_analysis()
    
    print(f"\n✓ Analysis complete for {n_rats} example rats")
    print(f"✓ Results saved to: {pipeline.save_path}")
    print(f"✓ Plots saved to: {pipeline.save_path}/aggregated_plots")
    
    return pipeline


def quick_example_analysis(model_type='betasort_test', verbose=True):
    """
    Quick function to analyze all 10 example rats.
    
    Parameters:
    -----------
    model_type : str
        Model type to use ('betasort', 'betasort_test', or 'betasort_OG')
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    BetasortPipeline : The pipeline object with results
    """
    return analyze_example_rats(model_type=model_type, verbose=verbose)


if __name__ == "__main__":
    # Example of new flexible workflow
    print("=== FLEXIBLE WORKFLOW EXAMPLE ===")
    
    #pipeline = analyze_example_rats(model_type="betasort")
    
    # 1. Analyze ALL rats (stores individual results)
    
    pipeline = BetasortPipeline(
        rats_to_exclude=['BP07', 'BP06', 'inferenceTesting'],
        model_type='betasort_test',
        use_diff_evolution=False
    )
    pipeline.run_analysis_only()
    
    #pipeline = BetasortPipeline.from_saved_data()
    pipeline.aggregate_and_plot(output_suffix="_VTE")
    
    # 2. Create different aggregated views
    #print("\n--- Creating plots excluding BP06-10 ---")
    #pipeline.aggregate_and_plot(rats_to_exclude=['BP06', 'BP07', 'BP08', 'BP09', 'BP10', 'BP13'], output_suffix="_OG_sim")
    
    