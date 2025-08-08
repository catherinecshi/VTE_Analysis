import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pointbiserialr
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class VTETransitionAnalyzer:
    def __init__(self, data_dir, correct_streak_length=5, incorrect_streak_length=2, exclude_rats=None):
        self.data_dir = Path(data_dir)
        self.correct_streak_length = correct_streak_length
        self.incorrect_streak_length = incorrect_streak_length
        self.exclude_rats = exclude_rats or []
        self.results = defaultdict(list)
        
    def extract_trial_number(self, id_str):
        """Extract trial number from ID string (integer after last underscore)"""
        return int(id_str.split('_')[-1])
    
    def sort_dataframe_by_trial(self, df):
        """Sort dataframe by trial number extracted from ID column"""
        df = df.copy()
        df['trial_number'] = df['ID'].apply(self.extract_trial_number)
        return df.sort_values('trial_number').reset_index(drop=True)
    
    def detect_correct_to_error_transitions(self, df, trial_type):
        """
        Detect transitions from correct streak to error for specific trial type
        Returns list of indices where transition occurs
        """
        trial_data = df[df['Trial_Type'] == trial_type].copy()
        if len(trial_data) < self.correct_streak_length + 1:
            return []
        
        transitions = []
        for i in range(len(trial_data) - self.correct_streak_length):
            # Check if we have correct_streak_length consecutive correct trials
            streak = trial_data.iloc[i:i+self.correct_streak_length]['Correct'].all()
            
            # Check if next trial is incorrect
            if i + self.correct_streak_length < len(trial_data):
                next_incorrect = not trial_data.iloc[i + self.correct_streak_length]['Correct']
                
                if streak and next_incorrect:
                    # Get original dataframe index of the error trial
                    error_trial_idx = trial_data.iloc[i + self.correct_streak_length].name
                    transitions.append(error_trial_idx)
        
        return transitions
    
    def detect_incorrect_to_correct_transitions(self, df, trial_type):
        """
        Detect transitions from incorrect streak to correct for specific trial type
        Returns list of indices where transition occurs
        """
        trial_data = df[df['Trial_Type'] == trial_type].copy()
        if len(trial_data) < self.incorrect_streak_length + 1:
            return []
        
        transitions = []
        for i in range(len(trial_data) - self.incorrect_streak_length):
            # Check if we have incorrect_streak_length consecutive incorrect trials
            streak = (~trial_data.iloc[i:i+self.incorrect_streak_length]['Correct']).all()
            
            # Check if next trial is correct
            if i + self.incorrect_streak_length < len(trial_data):
                next_correct = trial_data.iloc[i + self.incorrect_streak_length]['Correct']
                
                if streak and next_correct:
                    # Get original dataframe index of the correct trial
                    correct_trial_idx = trial_data.iloc[i + self.incorrect_streak_length].name
                    transitions.append(correct_trial_idx)
        
        return transitions
    
    def analyze_single_file(self, file_path):
        """Analyze single CSV file for transitions and VTE correlations"""
        df = pd.read_csv(file_path)
        df = self.sort_dataframe_by_trial(df)
        
        # Get unique trial types
        trial_types = df['Trial_Type'].unique()
        
        file_results = {
            'rat': file_path.parent.parent.name,
            'day': file_path.parent.name,
            'transitions': {}
        }
        
        for trial_type in trial_types:
            # Detect correct→error transitions
            correct_to_error = self.detect_correct_to_error_transitions(df, trial_type)
            
            # Detect incorrect→correct transitions  
            incorrect_to_correct = self.detect_incorrect_to_correct_transitions(df, trial_type)
            
            file_results['transitions'][trial_type] = {
                'correct_to_error': {
                    'transitions': correct_to_error,
                    'vte_on_transition': [df.loc[idx, 'VTE'] for idx in correct_to_error],
                    'vte_before_transition': []
                },
                'incorrect_to_correct': {
                    'transitions': incorrect_to_correct,
                    'vte_on_transition': [df.loc[idx, 'VTE'] for idx in incorrect_to_correct],
                    'vte_before_transition': []
                }
            }
            
            # Get VTE on trial before transition
            for idx in correct_to_error:
                if idx > 0:
                    file_results['transitions'][trial_type]['correct_to_error']['vte_before_transition'].append(
                        df.loc[idx-1, 'VTE']
                    )
            
            for idx in incorrect_to_correct:
                if idx > 0:
                    file_results['transitions'][trial_type]['incorrect_to_correct']['vte_before_transition'].append(
                        df.loc[idx-1, 'VTE']
                    )
        
        return file_results
    
    def process_all_files(self):
        """Process all CSV files matching the pattern"""
        pattern = str(self.data_dir / "*/*/zIdPhi_day_*.csv")
        files = glob.glob(pattern)
        
        # Filter out excluded rats
        if self.exclude_rats:
            original_count = len(files)
            files = [f for f in files if not any(rat in f for rat in self.exclude_rats)]
            excluded_count = original_count - len(files)
            print(f"Excluded {excluded_count} files from rats: {self.exclude_rats}")
        
        print(f"Found {len(files)} files to process")
        
        all_results = []
        for file_path in files:
            try:
                result = self.analyze_single_file(Path(file_path))
                all_results.append(result)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return all_results
    
    def calculate_correlations(self, all_results):
        """Calculate point-biserial correlations between transitions and VTE occurrence"""
        correlations = defaultdict(lambda: defaultdict(dict))
        
        # First, calculate overall VTE rates and total trials for each trial type across all files
        overall_vte_rates = defaultdict(list)
        total_trials_by_type = defaultdict(int)
        
        for result in all_results:
            # Load the original file to get all VTE data
            try:
                file_path = f"{self.data_dir}/{result['rat']}/{result['day']}/zIdPhi_day_{result['day']}.csv"
                df = pd.read_csv(file_path)
                df = self.sort_dataframe_by_trial(df)
                
                for trial_type in df['Trial_Type'].unique():
                    trial_data = df[df['Trial_Type'] == trial_type]
                    vte_values = [1 if vte else 0 for vte in trial_data['VTE']]
                    overall_vte_rates[trial_type].extend(vte_values)
                    total_trials_by_type[trial_type] += len(trial_data)
            except:
                continue
        
        # Calculate overall VTE rate for each trial type
        overall_rates = {}
        for trial_type, vte_list in overall_vte_rates.items():
            overall_rates[trial_type] = np.mean(vte_list)
            print(f"Overall VTE rate for Trial Type {trial_type}: {overall_rates[trial_type]:.3f} ({len(vte_list)} trials)")
        
        # Now calculate correlations
        for transition_type in ['correct_to_error', 'incorrect_to_correct']:
            for vte_timing in ['vte_on_transition', 'vte_before_transition']:
                # Collect data for each trial type
                trial_type_data = defaultdict(lambda: {'is_transition': [], 'vte_values': []})
                
                for result in all_results:
                    for trial_type, transitions in result['transitions'].items():
                        if trial_type in overall_rates:  # Make sure we have overall data
                            if transition_type in transitions:
                                trans_data = transitions[transition_type]
                                
                                # Add transition trials (is_transition = 1)
                                if len(trans_data['transitions']) > 0:
                                    if vte_timing in trans_data:
                                        vte_values = trans_data[vte_timing]
                                        for vte in vte_values:
                                            if vte is not None:
                                                trial_type_data[trial_type]['is_transition'].append(1)
                                                trial_type_data[trial_type]['vte_values'].append(1 if vte else 0)
                                
                                # Add non-transition trials (is_transition = 0) based on overall rates
                                # Sample non-transition trials to match transition count
                                n_transitions = len([v for v in trans_data.get(vte_timing, []) if v is not None])
                                if n_transitions > 0:
                                    # Add same number of non-transition samples
                                    overall_rate = overall_rates[trial_type]
                                    for _ in range(n_transitions * 2):  # 2x non-transitions for better comparison
                                        trial_type_data[trial_type]['is_transition'].append(0)
                                        # Sample VTE based on overall rate
                                        trial_type_data[trial_type]['vte_values'].append(1 if np.random.random() < overall_rate else 0)
                
                # Calculate correlations
                for trial_type, data in trial_type_data.items():
                    if len(data['is_transition']) >= 6:  # Need sufficient data
                        try:
                            correlation, p_value = pointbiserialr(data['is_transition'], data['vte_values'])
                            
                            # Calculate transition VTE rate
                            transition_indices = [i for i, x in enumerate(data['is_transition']) if x == 1]
                            transition_vte_rate = np.mean([data['vte_values'][i] for i in transition_indices])
                            
                            # Calculate proportion of transitions
                            total_trials = total_trials_by_type[trial_type]
                            transition_proportion = len(transition_indices) / total_trials if total_trials > 0 else 0
                            
                            correlations[transition_type][vte_timing][trial_type] = {
                                'correlation': correlation,
                                'p_value': p_value,
                                'n_transitions': len(transition_indices),
                                'vte_rate': transition_vte_rate,
                                'overall_vte_rate': overall_rates[trial_type],
                                'transition_proportion': transition_proportion,
                                'total_trials': total_trials
                            }
                            
                            print(f"{transition_type}, {vte_timing}, Trial Type {trial_type}: "
                                  f"correlation = {correlation:.3f} (p={p_value:.3f}), "
                                  f"transition VTE rate = {transition_vte_rate:.3f} vs overall = {overall_rates[trial_type]:.3f}")
                                  
                        except Exception as e:
                            print(f"Error calculating correlation for {transition_type}, {vte_timing}, trial_type {trial_type}: {e}")
        
        return correlations
    
    def analyze_streak_length_effects(self, all_results, max_streak_length=10):
        """Analyze how transition proportions and VTE correlations change with streak length"""
        streak_results = defaultdict(lambda: defaultdict(dict))
        
        # Calculate overall VTE rates for each trial type (same as before)
        overall_vte_rates = defaultdict(list)
        total_trials_by_type = defaultdict(int)
        
        for result in all_results:
            try:
                file_path = f"{self.data_dir}/{result['rat']}/{result['day']}/zIdPhi_day_{result['day']}.csv"
                df = pd.read_csv(file_path)
                df = self.sort_dataframe_by_trial(df)
                
                for trial_type in df['Trial_Type'].unique():
                    trial_data = df[df['Trial_Type'] == trial_type]
                    vte_values = [1 if vte else 0 for vte in trial_data['VTE']]
                    overall_vte_rates[trial_type].extend(vte_values)
                    total_trials_by_type[trial_type] += len(trial_data)
            except:
                continue
        
        overall_rates = {trial_type: np.mean(vte_list) for trial_type, vte_list in overall_vte_rates.items()}
        
        # Test different streak lengths
        for streak_len in range(2, max_streak_length + 1):
            print(f"Analyzing streak length: {streak_len}")
            
            # Temporarily change streak lengths
            original_correct = self.correct_streak_length
            original_incorrect = self.incorrect_streak_length
            
            # Use same streak length for both correct and incorrect for this analysis
            self.correct_streak_length = streak_len
            self.incorrect_streak_length = streak_len
            
            # Analyze transitions for this streak length (aggregate all data)
            streak_transitions = {
                'correct_to_error': {'vte_on_transition': {'transitions': [], 'vte_values': []}, 
                                   'vte_before_transition': {'transitions': [], 'vte_values': []}},
                'incorrect_to_correct': {'vte_on_transition': {'transitions': [], 'vte_values': []}, 
                                       'vte_before_transition': {'transitions': [], 'vte_values': []}}
            }
            
            for result in all_results:
                # Re-analyze file with new streak length
                try:
                    file_path = f"{self.data_dir}/{result['rat']}/{result['day']}/zIdPhi_day_{result['day']}.csv"
                    temp_result = self.analyze_single_file(Path(file_path))
                    
                    for trial_type, transitions in temp_result['transitions'].items():
                        if trial_type in overall_rates:
                            for transition_type in ['correct_to_error', 'incorrect_to_correct']:
                                if transition_type in transitions:
                                    trans_data = transitions[transition_type]
                                    
                                    # Collect transition data (aggregate across all trial types)
                                    if len(trans_data['transitions']) > 0:
                                        for vte_timing in ['vte_on_transition', 'vte_before_transition']:
                                            if vte_timing in trans_data:
                                                vte_values = trans_data[vte_timing]
                                                for vte in vte_values:
                                                    if vte is not None:
                                                        streak_transitions[transition_type][vte_timing]['transitions'].append(1)
                                                        streak_transitions[transition_type][vte_timing]['vte_values'].append(1 if vte else 0)
                                                        
                                                        # Add non-transition samples
                                                        overall_rate = overall_rates[trial_type]
                                                        for _ in range(2):  # 2x non-transitions
                                                            streak_transitions[transition_type][vte_timing]['transitions'].append(0)
                                                            streak_transitions[transition_type][vte_timing]['vte_values'].append(1 if np.random.random() < overall_rate else 0)
                except:
                    continue
            
            # Calculate statistics for this streak length (aggregate across all trial types)
            for transition_type in ['correct_to_error', 'incorrect_to_correct']:
                for vte_timing in ['vte_on_transition', 'vte_before_transition']:
                    data = streak_transitions[transition_type][vte_timing]
                    
                    if len(data['transitions']) >= 6:
                        # Calculate correlation
                        try:
                            correlation, p_value = pointbiserialr(data['transitions'], data['vte_values'])
                            
                            # Calculate transition proportion
                            n_transitions = sum(data['transitions'])
                            total_trials = sum(total_trials_by_type.values())
                            transition_proportion = n_transitions / total_trials if total_trials > 0 else 0
                            
                            # Calculate VTE rate during transitions
                            transition_indices = [i for i, x in enumerate(data['transitions']) if x == 1]
                            transition_vte_rate = np.mean([data['vte_values'][i] for i in transition_indices]) if transition_indices else 0
                            
                            streak_results[streak_len][f"{transition_type}_{vte_timing}"] = {
                                'correlation': correlation,
                                'p_value': p_value,
                                'transition_proportion': transition_proportion,
                                'n_transitions': n_transitions,
                                'vte_rate': transition_vte_rate
                            }
                            
                        except Exception as e:
                            print(f"Error for streak {streak_len}, {transition_type}, {vte_timing}: {e}")
            
            # Restore original streak lengths
            self.correct_streak_length = original_correct
            self.incorrect_streak_length = original_incorrect
        
        return dict(streak_results)
    
    def plot_streak_length_analysis(self, streak_results):
        """Create plots showing how metrics change with streak length"""
        if not streak_results:
            print("No streak length data available for plotting")
            return
        
        # Prepare data
        streak_lengths = sorted(streak_results.keys())
        plot_data = []
        
        for streak_len in streak_lengths:
            for condition, stats in streak_results[streak_len].items():
                # Parse condition name
                parts = condition.split('_')
                if len(parts) >= 4:
                    transition_type = f"{parts[0]} → {parts[2]}"  # correct_to_error -> Correct → Error
                    vte_timing = 'On Transition' if 'on' in condition else 'Before Transition'
                    
                    plot_data.append({
                        'Streak_Length': streak_len,
                        'Transition_Type': transition_type.title(),
                        'VTE_Timing': vte_timing,
                        'Correlation': stats['correlation'],
                        'Transition_Proportion': stats['transition_proportion'],
                        'VTE_Rate': stats['vte_rate']
                    })
        
        if not plot_data:
            print("No data available for streak length plotting")
            return
        
        df_streak = pd.DataFrame(plot_data)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Streak Length Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Transition Proportion vs Streak Length
        ax1 = axes[0]
        try:
            for transition_type in df_streak['Transition_Type'].unique():
                data = df_streak[df_streak['Transition_Type'] == transition_type]
                avg_props = data.groupby('Streak_Length')['Transition_Proportion'].mean()
                ax1.plot(avg_props.index, avg_props.values, marker='o', label=transition_type)
        except Exception as e:
            print(f"Error plotting transition proportions: {e}")
        
        ax1.set_xlabel('Streak Length')
        ax1.set_ylabel('Transition Proportion')
        ax1.set_title('Transition Proportion vs Streak Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correlation vs Streak Length (On Transition only)
        ax2 = axes[1]
        try:
            for transition_type in df_streak['Transition_Type'].unique():
                # Use only "On Transition" timing
                data = df_streak[(df_streak['Transition_Type'] == transition_type) & 
                               (df_streak['VTE_Timing'] == 'On Transition')]
                if len(data) > 0:
                    ax2.plot(data['Streak_Length'], data['Correlation'], 
                           marker='o', label=transition_type, linewidth=2)
        except Exception as e:
            print(f"Error plotting correlations: {e}")
        
        ax2.set_xlabel('Streak Length')
        ax2.set_ylabel('Point-Biserial Correlation')
        ax2.set_title('VTE Correlation vs Streak Length')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('streak_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n=== STREAK LENGTH ANALYSIS SUMMARY ===")
        for streak_len in sorted(streak_results.keys()):
            print(f"\nStreak Length {streak_len}:")
            for condition, stats in streak_results[streak_len].items():
                print(f"  {condition}: correlation = {stats['correlation']:.3f}, "
                      f"proportion = {stats['transition_proportion']:.4f}, "
                      f"n_transitions = {stats['n_transitions']}")

        return streak_results
    
    def create_plots(self, correlations):
        """Create visualization plots"""
        # Prepare data for plotting
        plot_data = []
        
        for transition_type, timing_data in correlations.items():
            for vte_timing, trial_data in timing_data.items():
                for trial_type, stats in trial_data.items():
                    plot_data.append({
                        'Transition_Type': transition_type.replace('_', ' → ').title(),
                        'VTE_Timing': 'On Transition' if vte_timing == 'vte_on_transition' else 'Before Transition',
                        'Trial_Type': f'Trial Type {trial_type}',
                        'Correlation': stats['correlation'],
                        'P_Value': stats['p_value'],
                        'N_Transitions': stats['n_transitions'],
                        'VTE_Rate': stats['vte_rate'],
                        'Transition_Proportion': stats['transition_proportion'],
                        'Total_Trials': stats['total_trials']
                    })
        
        if not plot_data:
            print("No data available for plotting")
            return
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('VTE-Performance Transition Correlations', fontsize=16, fontweight='bold')
        
        # Plot 1: Correlation by transition type and VTE timing (without outliers)
        ax1 = axes[0, 0]
        try:
            sns.boxplot(data=df_plot, x='Transition_Type', y='Correlation', hue='VTE_Timing', ax=ax1, showfliers=False)
        except (UnboundLocalError, AttributeError, ValueError):
            # Fallback to matplotlib if seaborn has compatibility issues
            for i, timing in enumerate(['On Transition', 'Before Transition']):
                data = df_plot[df_plot['VTE_Timing'] == timing]
                if len(data) > 0:
                    positions = [j + i*0.4 for j in range(len(data['Transition_Type'].unique()))]
                    try:
                        ax1.boxplot([data[data['Transition_Type'] == tt]['Correlation'].values 
                                   for tt in data['Transition_Type'].unique()], 
                                  positions=positions, widths=0.3, patch_artist=True)
                    except:
                        # Simple scatter if boxplot fails
                        for j, tt in enumerate(data['Transition_Type'].unique()):
                            vals = data[data['Transition_Type'] == tt]['Correlation'].values
                            ax1.scatter([j + i*0.4] * len(vals), vals)
            ax1.set_xticks(range(len(df_plot['Transition_Type'].unique())))
            ax1.set_xticklabels(df_plot['Transition_Type'].unique())
        
        ax1.set_title('Correlation by Transition Type')
        ax1.set_ylabel('Point-Biserial Correlation')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Correlation by trial type (ordered, no outliers)
        ax2 = axes[0, 1]
        # Order trial types numerically
        trial_type_order = sorted(df_plot['Trial_Type'].unique(), key=lambda x: int(x.split()[-1]))
        
        try:
            sns.boxplot(data=df_plot, x='Trial_Type', y='Correlation', ax=ax2, order=trial_type_order, showfliers=False)
        except (UnboundLocalError, AttributeError, ValueError):
            # Fallback to matplotlib
            try:
                ax2.boxplot([df_plot[df_plot['Trial_Type'] == tt]['Correlation'].values 
                            for tt in trial_type_order], showfliers=False)
                ax2.set_xticklabels(trial_type_order)
            except:
                # Simple scatter if boxplot fails
                for j, tt in enumerate(trial_type_order):
                    vals = df_plot[df_plot['Trial_Type'] == tt]['Correlation'].values
                    ax2.scatter([j+1] * len(vals), vals)
                ax2.set_xticks(range(1, len(trial_type_order)+1))
                ax2.set_xticklabels(trial_type_order)
        
        ax2.set_title('Correlation by Trial Type')
        ax2.set_ylabel('Point-Biserial Correlation')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: VTE Rate during transitions with SEM error bars
        ax3 = axes[1, 0]
        try:
            sns.barplot(data=df_plot, x='Transition_Type', y='VTE_Rate', hue='VTE_Timing', ax=ax3, capsize=0.1)
        except (UnboundLocalError, AttributeError, ValueError):
            # Fallback to matplotlib with SEM error bars
            transition_types = df_plot['Transition_Type'].unique()
            width = 0.35
            x = np.arange(len(transition_types))
            
            # Calculate means and SEM for each group
            on_trans_data = df_plot[df_plot['VTE_Timing'] == 'On Transition']
            before_trans_data = df_plot[df_plot['VTE_Timing'] == 'Before Transition']
            
            on_trans_means = []
            on_trans_sems = []
            before_trans_means = []
            before_trans_sems = []
            
            for tt in transition_types:
                on_vals = on_trans_data[on_trans_data['Transition_Type'] == tt]['VTE_Rate'].values
                before_vals = before_trans_data[before_trans_data['Transition_Type'] == tt]['VTE_Rate'].values
                
                on_trans_means.append(np.mean(on_vals) if len(on_vals) > 0 else 0)
                on_trans_sems.append(np.std(on_vals) / np.sqrt(len(on_vals)) if len(on_vals) > 1 else 0)
                
                before_trans_means.append(np.mean(before_vals) if len(before_vals) > 0 else 0)
                before_trans_sems.append(np.std(before_vals) / np.sqrt(len(before_vals)) if len(before_vals) > 1 else 0)
            
            ax3.bar(x - width/2, on_trans_means, width, yerr=on_trans_sems, capsize=5, label='On Transition')
            ax3.bar(x + width/2, before_trans_means, width, yerr=before_trans_sems, capsize=5, label='Before Transition')
            ax3.set_xticks(x)
            ax3.set_xticklabels(transition_types)
            ax3.legend()
        
        ax3.set_title('VTE Rate During Transitions')
        ax3.set_ylabel('VTE Rate')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Proportion of transitions by type (ordered, no error bars)
        ax4 = axes[1, 1]
        # Order trial types numerically
        trial_type_order_4 = sorted(df_plot['Trial_Type'].unique(), key=lambda x: int(x.split()[-1]))
        
        try:
            sns.barplot(data=df_plot, x='Trial_Type', y='Transition_Proportion', hue='Transition_Type', ax=ax4, order=trial_type_order_4)
        except (UnboundLocalError, AttributeError, ValueError):
            # Fallback to matplotlib (no error bars)
            width = 0.35
            x = np.arange(len(trial_type_order_4))
            
            # Calculate means for transition proportions
            correct_to_error_data = df_plot[df_plot['Transition_Type'] == 'Correct → Error']
            incorrect_to_correct_data = df_plot[df_plot['Transition_Type'] == 'Incorrect → Correct']
            
            c2e_means = []
            i2c_means = []
            
            for tt in trial_type_order_4:
                c2e_vals = correct_to_error_data[correct_to_error_data['Trial_Type'] == tt]['Transition_Proportion'].values
                i2c_vals = incorrect_to_correct_data[incorrect_to_correct_data['Trial_Type'] == tt]['Transition_Proportion'].values
                
                c2e_means.append(np.mean(c2e_vals) if len(c2e_vals) > 0 else 0)
                i2c_means.append(np.mean(i2c_vals) if len(i2c_vals) > 0 else 0)
            
            ax4.bar(x - width/2, c2e_means, width, label='Correct → Error')
            ax4.bar(x + width/2, i2c_means, width, label='Incorrect → Correct')
            ax4.set_xticks(x)
            ax4.set_xticklabels(trial_type_order_4)
            ax4.legend()
        
        ax4.set_title('Proportion of Transitions by Type')
        ax4.set_ylabel('Transition Proportion')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('vte_transition_analysis_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n=== CORRELATION SUMMARY ===")
        for _, row in df_plot.iterrows():
            print(f"{row['Transition_Type']} ({row['VTE_Timing']} - {row['Trial_Type']}):")
            print(f"  Correlation: {row['Correlation']:.3f} (p={row['P_Value']:.3f})")
            print(f"  N transitions: {row['N_Transitions']}, VTE rate: {row['VTE_Rate']:.3f}")
            print()

def main():
    # Initialize analyzer
    data_dir = "/Users/catpillow/Documents/VTE_Analysis/processed_data/VTE_values"
    exclude_rats = ['BP06', 'BP07', 'BP08', 'BP09', 'BP10']
    
    analyzer = VTETransitionAnalyzer(
        data_dir, 
        correct_streak_length=5, 
        incorrect_streak_length=2,
        exclude_rats=exclude_rats
    )
    
    # Process all files
    print("Processing all data files...")
    all_results = analyzer.process_all_files()
    
    # Calculate correlations for original streak lengths
    print("Calculating correlations...")
    correlations = analyzer.calculate_correlations(all_results)
    
    # Create main plots
    print("Creating main visualization...")
    analyzer.create_plots(correlations)
    
    # Analyze streak length effects
    print("Analyzing streak length effects...")
    streak_results = analyzer.analyze_streak_length_effects(all_results, max_streak_length=8)
    
    # Create streak length plots
    print("Creating streak length analysis plots...")
    analyzer.plot_streak_length_analysis(streak_results)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()