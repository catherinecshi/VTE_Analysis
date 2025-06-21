import os
import numpy as np
import pandas as pd

from models import betasort

# Main analysis
data_path = os.path.join(helper.BASE_PATH, "processed_data", "data_for_model")
save_path = os.path.join(helper.BASE_PATH, "processed_data", "vte_analysis")

for rat in os.listdir(data_path):
    if "TH405" not in rat:  # Your existing filter
        continue

    rat_path = os.path.join(data_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                continue

            file_path = os.path.join(root, file)
            file_csv = pd.read_csv(file_path)
            
            # Run pair-specific VTE analysis
            pair_vte_df, all_models = betasort.analyze_vte_uncertainty(
                file_csv, rat, tau=0.05, xi=0.95
            )
            
            # Analyze correlations
            correlation_results = betasort.analyze_correlations(pair_vte_df)
            
            # Create output directory
            rat_output_dir = os.path.join(save_path, rat)
            os.makedirs(rat_output_dir, exist_ok=True)
            
            # Save results
            pair_vte_df.to_csv(os.path.join(rat_output_dir, f"{rat}_pair_vte_data.csv"), index=False)
            
            # Generate visualizations
            betasort.plot_vte_uncertainty(pair_vte_df, correlation_results, rat_output_dir)
            
            # Print summary of results
            print(f"Results for {rat}:")
            print("Overall correlations between VTE and uncertainty measures:")
            for measure, result in correlation_results['overall'].items():
                if isinstance(result, dict):
                    print(f"  {measure}: r={result['r']:.3f}, p={result['p']:.3f}")
            
            print("\nCorrelations by uncertainty measure (VTE vs non-VTE trials):")
            for measure, result in correlation_results['by_uncertainty_measure'].items():
                print(f"  {measure}: diff={result['difference']:.3f}, t={result['t_stat']:.3f}, p={result['p_value']:.3f}")
            
            print("\nPair-specific significant correlations:")
            for pair, pair_results in correlation_results['by_pair'].items():
                if isinstance(pair_results, dict):
                    for measure in ['stim1_uncertainty', 'stim2_uncertainty', 'pair_relational_uncertainty', 'pair_roc_uncertainty']:
                        if measure in pair_results and isinstance(pair_results[measure], dict) and pair_results[measure]['p'] < 0.05:
                            print(f"  Pair {pair}, {measure}: r={pair_results[measure]['r']:.3f}, p={pair_results[measure]['p']:.3f}")