#!/usr/bin/env python3
"""
Neural Update Rule Discovery Script for Betasort Model

This script learns a universal update rule for the Betasort model by training
a neural network to maximize choice prediction accuracy across multiple rats.

The goal is to discover what the optimal cognitive update process should be,
rather than using our human hypothesis about how rats learn.

Usage:
    python train_neural_update_rule.py --config config.yaml
    
Or run with default settings:
    python train_neural_update_rule.py
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the neural update rule classes (assuming they're in a separate module)
from models.betasort_nn import UniversalUpdateRule, NeuralBetasort, train_universal_update_rule
from models.betasort import Betasort  # Your original model
from config.paths import paths  # Your existing path configuration
from utilities import logging_utils

# Set up logging to track the experimental process
logger = logging_utils.setup_script_logger()

class NeuralUpdateExperiment:
    """
    Main experimental class that orchestrates the neural update rule discovery process.
    
    This class handles data loading, training coordination, evaluation, and result saving.
    Think of it as your experimental protocol - it ensures everything runs in the right
    order and that results are properly documented.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the experiment with configuration parameters
        
        Args:
            config: Dictionary containing all experimental parameters
        """
        self.config = config
        self.data_path = Path(config.get('data_path', paths.preprocessed_data_model))
        self.output_path = Path(config.get('output_path', paths.betasort_data)) / 'neural_update_rule'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        (self.output_path / 'models').mkdir(exist_ok=True)
        (self.output_path / 'results').mkdir(exist_ok=True)
        (self.output_path / 'visualizations').mkdir(exist_ok=True)
        
        # Training parameters
        self.epochs = config.get('epochs', 500)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.validation_split = config.get('validation_split', 0.2)
        self.n_simulations = config.get('n_simulations', 100)
        
        # Experiment tracking
        self.experiment_id = f"neural_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {
            'experiment_id': self.experiment_id,
            'config': config,
            'training_history': {},
            'evaluation_results': {},
            'learned_rule_analysis': {}
        }
        
        logger.info(f"Initialized experiment {self.experiment_id}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output path: {self.output_path}")
    
    def load_rat_data(self, rats_to_include: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and consolidate data from multiple rats
        
        This function replicates your existing data loading pattern but consolidates
        everything into a single DataFrame for neural network training.
        
        Args:
            rats_to_include: List of rat IDs to include. If None, includes all rats.
            
        Returns:
            Consolidated DataFrame with all rat data
        """
        logger.info("Loading rat data...")
        
        all_data = []
        processed_rats = []
        
        # Iterate through rat directories (matching your existing pattern)
        for rat_dir in os.listdir(self.data_path):
            # Apply rat filtering if specified
            if rats_to_include and rat_dir not in rats_to_include:
                continue
                
            # Skip system files (matching your existing logic)
            if rat_dir.startswith('.'):
                continue
                
            rat_path = os.path.join(self.data_path, rat_dir)
            if not os.path.isdir(rat_path):
                continue
            
            logger.info(f"Processing rat {rat_dir}...")
            
            # Find the consolidated data file (matching your "all_days" pattern)
            rat_data_found = False
            for root, dirs, files in os.walk(rat_path):
                for file in files:
                    # Skip system files and non-consolidated files
                    if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        # Load the CSV data
                        rat_df = pd.read_csv(file_path)
                        
                        # Add rat identifier column
                        rat_df['rat'] = rat_dir
                        
                        # Validate required columns
                        required_cols = ['Day', 'first', 'second', 'correct']
                        missing_cols = [col for col in required_cols if col not in rat_df.columns]
                        if missing_cols:
                            logger.warning(f"Rat {rat_dir} missing columns: {missing_cols}")
                            continue
                        
                        # Add to consolidated dataset
                        all_data.append(rat_df)
                        processed_rats.append(rat_dir)
                        rat_data_found = True
                        
                        logger.info(f"  Loaded {len(rat_df)} trials across {rat_df['Day'].nunique()} days")
                        break
                    except Exception as e:
                        logger.error(f"error when loading data for {rat_dir}: {e}")
                        
                if rat_data_found:
                    break
            
            if not rat_data_found:
                logger.warning(f"No valid data files found for rat {rat_dir}")
        
        if not all_data:
            raise ValueError("No valid rat data found! Check your data path and file format.")
        
        # Consolidate all rat data
        consolidated_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"Successfully loaded data from {len(processed_rats)} rats:")
        logger.info(f"  Rats: {processed_rats}")
        logger.info(f"  Total trials: {len(consolidated_df)}")
        logger.info(f"  Total days: {consolidated_df.groupby('rat')['Day'].nunique().sum()}")
        
        # Save the consolidated dataset for reproducibility
        consolidated_path = self.output_path / 'consolidated_data.csv'
        consolidated_df.to_csv(consolidated_path, index=False)
        logger.info(f"Saved consolidated data to {consolidated_path}")
        
        return consolidated_df
    
    def train_neural_update_rule(self, data_df: pd.DataFrame) -> UniversalUpdateRule:
        """
        Train the neural network to learn the optimal update rule
        
        This is where the magic happens - we're essentially reverse-engineering
        the cognitive process that makes rat choices most predictable.
        
        Args:
            data_df: Consolidated rat data
            
        Returns:
            Trained neural update rule
        """
        logger.info("Starting neural update rule training...")
        logger.info(f"Training parameters:")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Validation split: {self.validation_split}")
        
        # Train the universal update rule
        # This function handles all the complex training logic
        trained_rule = train_universal_update_rule(
            data_df,
            rats_to_include=None,  # Use all rats in the consolidated data
            epochs=self.epochs,
            learning_rate=self.learning_rate
        )
        
        # Save the trained model
        model_path = self.output_path / 'models' / f'{self.experiment_id}_neural_update_rule.pth'
        torch.save(trained_rule.state_dict(), model_path)
        logger.info(f"Saved trained model to {model_path}")
        
        # Analyze what the network learned
        interpretation = trained_rule.interpret_learned_rule()
        self.results['learned_rule_analysis'] = interpretation
        
        logger.info("Neural update rule training completed!")
        logger.info("Learned rule analysis:")
        logger.info(f"  Global update scale: {interpretation['global_scale']:.3f}")
        logger.info(f"  Strongest parameter bias: {max(interpretation['parameter_biases'].items(), key=lambda x: abs(x[1]))}")
        logger.info(f"  Most important feature: {max(interpretation['feature_importance'].items(), key=lambda x: x[1])}")
        
        return trained_rule
    
    def evaluate_models(self, data_df: pd.DataFrame, trained_rule: UniversalUpdateRule) -> Dict:
        """
        Compare the original Betasort model with the neural version
        
        This evaluation tells us how much better we can predict rat choices
        using the learned update rule versus our original hypothesis.
        
        Args:
            data_df: Test data
            trained_rule: Trained neural update rule
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Evaluating model performance...")
        
        evaluation_results = {
            'per_rat_results': {},
            'overall_performance': {},
            'improvement_analysis': {}
        }
        
        rats = data_df['rat'].unique()
        all_original_matches = []
        all_neural_matches = []
        
        for rat in rats:
            logger.info(f"Evaluating rat {rat}...")
            
            rat_data = data_df[data_df['rat'] == rat]
            rat_results = self._evaluate_single_rat(rat_data, rat, trained_rule)
            
            evaluation_results['per_rat_results'][rat] = rat_results
            all_original_matches.extend(rat_results['original_matches'])
            all_neural_matches.extend(rat_results['neural_matches'])
        
        # Calculate overall performance metrics
        original_mean = np.mean(all_original_matches)
        neural_mean = np.mean(all_neural_matches)
        improvement = neural_mean - original_mean
        percent_improvement = (improvement / original_mean) * 100 if original_mean > 0 else 0
        
        evaluation_results['overall_performance'] = {
            'original_accuracy': original_mean,
            'neural_accuracy': neural_mean,
            'absolute_improvement': improvement,
            'percent_improvement': percent_improvement
        }
        
        # Statistical significance testing
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(all_neural_matches, all_original_matches)
        evaluation_results['statistical_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        logger.info("Evaluation completed!")
        logger.info(f"Original model accuracy: {original_mean:.3f}")
        logger.info(f"Neural model accuracy: {neural_mean:.3f}")
        logger.info(f"Improvement: +{improvement:.3f} ({percent_improvement:+.1f}%)")
        logger.info(f"Statistical significance: p = {p_value:.3f}")
        
        self.results['evaluation_results'] = evaluation_results
        return evaluation_results
    
    def _evaluate_single_rat(self, rat_data: pd.DataFrame, rat_id: str, trained_rule: UniversalUpdateRule) -> Dict:
        """
        Evaluate both models on a single rat's data
        
        This function runs both the original and neural models through the exact
        same sequence of trials and compares their choice prediction accuracy.
        
        Args:
            rat_data: Data for a single rat
            rat_id: Rat identifier
            trained_rule: Trained neural update rule
            
        Returns:
            Dictionary with evaluation results for this rat
        """
        original_matches = []
        neural_matches = []
        
        # Track global state across days (matching your existing approach)
        global_U, global_L, global_R, global_N = {}, {}, {}, {}
        neural_global_U, neural_global_L, neural_global_R, neural_global_N = {}, {}, {}, {}
        
        for day, day_data in rat_data.groupby('Day'):
            chosen_trials = day_data['first'].values
            unchosen_trials = day_data['second'].values
            
            # Determine stimuli present on this day
            all_stimuli = set(np.concatenate([chosen_trials, unchosen_trials]))
            n_stimuli = max(all_stimuli) + 1
            
            # Initialize both models for this day
            original_model = Betasort(n_stimuli, rat_id, day)
            neural_model = NeuralBetasort(n_stimuli, rat_id, day, trained_rule)
            
            # Transfer previous state to both models
            for stim_idx in range(n_stimuli):
                if stim_idx in global_U:
                    original_model.U[stim_idx] = global_U[stim_idx]
                    original_model.L[stim_idx] = global_L[stim_idx]
                    original_model.R[stim_idx] = global_R[stim_idx]
                    original_model.N[stim_idx] = global_N[stim_idx]
                    
                    neural_model.U[stim_idx] = neural_global_U[stim_idx]
                    neural_model.L[stim_idx] = neural_global_L[stim_idx]
                    neural_model.R[stim_idx] = neural_global_R[stim_idx]
                    neural_model.N[stim_idx] = neural_global_N[stim_idx]
            
            # Test both models on each trial
            for t in range(len(chosen_trials)):
                chosen = chosen_trials[t]
                unchosen = unchosen_trials[t]
                actual_choice = chosen
                
                # Simulate choices with both models
                original_choices = [original_model.choose([chosen, unchosen]) for _ in range(self.n_simulations)]
                neural_choices = [neural_model.choose([chosen, unchosen]) for _ in range(self.n_simulations)]
                
                # Calculate match rates with actual rat choice
                original_match_rate = np.mean([c == actual_choice for c in original_choices])
                neural_match_rate = np.mean([c == actual_choice for c in neural_choices])
                
                original_matches.append(original_match_rate)
                neural_matches.append(neural_match_rate)
                
                # Update both models based on actual outcome
                reward = 1 if chosen < unchosen else 0
                original_model.update(chosen, unchosen, reward, original_match_rate, 0.6)
                neural_model.update(chosen, unchosen, reward)
            
            # Save final states for next day
            for stim_idx in range(n_stimuli):
                global_U[stim_idx] = original_model.U[stim_idx]
                global_L[stim_idx] = original_model.L[stim_idx]
                global_R[stim_idx] = original_model.R[stim_idx]
                global_N[stim_idx] = original_model.N[stim_idx]
                
                neural_global_U[stim_idx] = neural_model.U[stim_idx]
                neural_global_L[stim_idx] = neural_model.L[stim_idx]
                neural_global_R[stim_idx] = neural_model.R[stim_idx]
                neural_global_N[stim_idx] = neural_model.N[stim_idx]
        
        return {
            'original_matches': original_matches,
            'neural_matches': neural_matches,
            'original_mean': np.mean(original_matches),
            'neural_mean': np.mean(neural_matches),
            'improvement': np.mean(neural_matches) - np.mean(original_matches)
        }
    
    def create_visualizations(self, data_df: pd.DataFrame, trained_rule: UniversalUpdateRule):
        """
        Create comprehensive visualizations of the results
        
        These visualizations help us understand what the neural network learned
        and how it differs from our original hypothesis about rat cognition.
        """
        logger.info("Creating visualizations...")
        
        viz_path = self.output_path / 'visualizations'
        
        # 1. Model comparison across rats
        self._plot_model_comparison(viz_path / 'model_comparison.png')
        
        # 2. Learned rule interpretation
        self._plot_learned_rule_analysis(trained_rule, viz_path / 'learned_rule_analysis.png')
        
        # 3. Feature importance analysis
        self._plot_feature_importance(trained_rule, viz_path / 'feature_importance.png')
        
        # 4. Per-rat performance analysis
        self._plot_per_rat_performance(viz_path / 'per_rat_performance.png')
        
        logger.info(f"Visualizations saved to {viz_path}")
    
    def _plot_model_comparison(self, save_path: Path):
        """Create comparison plot between original and neural models"""
        eval_results = self.results['evaluation_results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall performance comparison
        original_acc = eval_results['overall_performance']['original_accuracy']
        neural_acc = eval_results['overall_performance']['neural_accuracy']
        
        ax1.bar(['Original Model', 'Neural Model'], [original_acc, neural_acc], 
                color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Choice Prediction Accuracy')
        ax1.set_title('Overall Model Performance')
        ax1.set_ylim(0, 1)
        
        # Per-rat comparison
        rats = list(eval_results['per_rat_results'].keys())
        original_means = [eval_results['per_rat_results'][rat]['original_mean'] for rat in rats]
        neural_means = [eval_results['per_rat_results'][rat]['neural_mean'] for rat in rats]
        
        x_pos = np.arange(len(rats))
        width = 0.35
        
        ax2.bar(x_pos - width/2, original_means, width, label='Original', color='skyblue')
        ax2.bar(x_pos + width/2, neural_means, width, label='Neural', color='lightcoral')
        ax2.set_xlabel('Rat')
        ax2.set_ylabel('Choice Prediction Accuracy')
        ax2.set_title('Per-Rat Performance Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(rats, rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learned_rule_analysis(self, trained_rule: UniversalUpdateRule, save_path: Path):
        """Visualize what the neural network learned about the update process"""
        interpretation = trained_rule.interpret_learned_rule()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Parameter biases
        biases = interpretation['parameter_biases']
        ax1.bar(biases.keys(), biases.values(), color='lightgreen')
        ax1.set_title('Learned Parameter Update Biases')
        ax1.set_ylabel('Bias Magnitude')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pathway strength comparison
        reward_strength = interpretation['reward_pathway_strength']
        no_reward_strength = interpretation['no_reward_pathway_strength']
        ax2.bar(['Reward Pathway', 'No-Reward Pathway'], 
                [reward_strength, no_reward_strength], 
                color=['gold', 'silver'])
        ax2.set_title('Pathway Strength Analysis')
        ax2.set_ylabel('Average Weight Magnitude')
        
        # Feature importance (top 8)
        importance = interpretation['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
        features, values = zip(*sorted_features)
        
        ax3.barh(range(len(features)), values, color='lightblue')
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels(features)
        ax3.set_xlabel('Importance Score')
        ax3.set_title('Top Feature Importances')
        
        # Global scaling visualization
        global_scale = interpretation['global_scale']
        ax4.text(0.5, 0.5, f'Global Update Scale\n{global_scale:.3f}', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle='round', facecolor='wheat'))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Learned Scaling Factor')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, trained_rule: UniversalUpdateRule, save_path: Path):
        """Detailed feature importance analysis"""
        interpretation = trained_rule.interpret_learned_rule()
        importance = interpretation['feature_importance']
        
        # Sort all features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), values, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance Score')
        plt.title('Complete Feature Importance Analysis\n(What influences the learned update rule most?)')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_rat_performance(self, save_path: Path):
        """Detailed per-rat performance analysis"""
        eval_results = self.results['evaluation_results']['per_rat_results']
        
        rats = list(eval_results.keys())
        improvements = [eval_results[rat]['improvement'] for rat in rats]
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if imp < 0 else 'green' for imp in improvements]
        bars = plt.bar(rats, improvements, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Rat')
        plt.ylabel('Improvement (Neural - Original)')
        plt.title('Per-Rat Performance Improvement\n(Positive = Neural model better)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height >= 0 else height - 0.01,
                    f'{improvement:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all experimental results and analysis"""
        results_path = self.output_path / 'results' / f'{self.experiment_id}_results.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Also save a human-readable summary
        summary_path = self.output_path / 'results' / f'{self.experiment_id}_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Neural Update Rule Discovery Experiment\n")
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL RESULTS:\n")
            f.write(f"Original model accuracy: {self.results['evaluation_results']['overall_performance']['original_accuracy']:.3f}\n")
            f.write(f"Neural model accuracy: {self.results['evaluation_results']['overall_performance']['neural_accuracy']:.3f}\n")
            f.write(f"Improvement: {self.results['evaluation_results']['overall_performance']['absolute_improvement']:.3f} ({self.results['evaluation_results']['overall_performance']['percent_improvement']:+.1f}%)\n")
            f.write(f"Statistical significance: p = {self.results['evaluation_results']['statistical_test']['p_value']:.4f}\n\n")
            
            f.write("LEARNED RULE ANALYSIS:\n")
            analysis = self.results['learned_rule_analysis']
            f.write(f"Global update scale: {analysis['global_scale']:.3f}\n")
            f.write(f"Parameter biases: {analysis['parameter_biases']}\n")
            f.write(f"Most important features:\n")
            sorted_features = sorted(analysis['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                f.write(f"  {feature}: {importance:.3f}\n")
        
        logger.info(f"Summary saved to {summary_path}")
    
    def run_complete_experiment(self, rats_to_include: Optional[List[str]] = None):
        """
        Run the complete neural update rule discovery experiment
        
        This is the main experimental pipeline that orchestrates all steps
        """
        logger.info(f"Starting complete neural update rule discovery experiment")
        logger.info(f"Experiment ID: {self.experiment_id}")
        
        try:
            # Step 1: Load and consolidate data
            data_df = self.load_rat_data(rats_to_include)
            
            # Step 2: Train the neural update rule
            trained_rule = self.train_neural_update_rule(data_df)
            
            # Step 3: Evaluate both models
            evaluation_results = self.evaluate_models(data_df, trained_rule)
            
            # Step 4: Create visualizations
            self.create_visualizations(data_df, trained_rule)
            
            # Step 5: Save all results
            self.save_results()
            
            logger.info("Experiment completed successfully!")
            logger.info(f"Results saved to: {self.output_path}")
            
            # Print summary to console
            overall_perf = evaluation_results['overall_performance']
            logger.info("\n" + "="*50)
            logger.info("EXPERIMENT SUMMARY")
            logger.info("="*50)
            logger.info(f"Original Model Accuracy: {overall_perf['original_accuracy']:.3f}")
            logger.info(f"Neural Model Accuracy: {overall_perf['neural_accuracy']:.3f}")
            logger.info(f"Improvement: +{overall_perf['absolute_improvement']:.3f} ({overall_perf['percent_improvement']:+.1f}%)")
            logger.info(f"Statistical significance: p = {evaluation_results['statistical_test']['p_value']:.4f}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load experimental configuration"""
    default_config = {
        'epochs': 500,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'n_simulations': 100,
        'rats_to_include': None,  # None means include all rats
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config

def main():
    """Main entry point for the neural update rule discovery script"""
    parser = argparse.ArgumentParser(
        description='Discover optimal update rule for Betasort model using neural networks'
    )
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON)')
    parser.add_argument('--rats', nargs='+', help='Specific rats to include (e.g., TH510 TH511)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.rats:
        config['rats_to_include'] = args.rats
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Run the experiment
    experiment = NeuralUpdateExperiment(config)
    experiment.run_complete_experiment(config.get('rats_to_include'))

if __name__ == "__main__":
    main()