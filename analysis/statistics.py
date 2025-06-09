import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

def one_way_anova(data_groups: list) -> dict:
    """
    One way ANOVA
    
    Parameters:
        - data_groups (list of arrays): List containing arrays of data for each group
    
    Returns:
        - dict
            - "f_stat": F-statistic from ANOVA
            - "p_value": p-value from ANOVA
    """
    
    # one way ANOVA
    f_stat, p_value = stats.f_oneway(*data_groups)
    
    return {
        "f_stat": f_stat,
        "p_value": p_value
    }

def plot_one_way_anova_bar(data_groups, group_labels=None, title=None, xlabel=None, ylabel=None):
    """
    Creates a bar plot with error bars and significance indicators
    
    Parameters:
        - data_groups (list of arrays): arrays of data for each group
        - group_labels (str list): (optional) labels for each group
        - title (str): (optional)
        - xlabel (str): (optional)
        - ylabel (str): (optional)
    
    Returns:
        - fig: plt figure object
        - ax: plt axis object
        - stats_results (dict): results from one_way_anova
    """
    
    stats_results = one_way_anova(data_groups)
    
    # means & SEM
    means = [np.mean(group) * 100 for group in data_groups]
    sems = [stats.sem(group) * 100 for group in data_groups]
    
    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data_groups))
    
    ax.bar(x, means, yerr=sems, capsize=5)
    
    # add plot elements
    if group_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
    
    if title:
        ax.set_title(title)
    
    if xlabel:
        ax.set_xlabel(xlabel)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # add anova results to plot
    anova_text = f"One-way ANOVA: \nF = {stats_results["f_stat"]:.3f}\np = {stats_results["p_value"]:.2e}"
    ax.text(0.95, 0.95, anova_text, transform=ax.transAxes, verticalalignment="top",
            horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig, ax, stats_results

def plot_one_way_anova_line(data_groups, group_labels=None, title=None, xlabel=None, ylabel=None):
    """
    Creates a line plot with error bars and ANOVA statistics
    
    Parameters:
        - data_groups (list of arrays): arrays of data for each group
        - group_labels (str list): (optional) labels for each group
        - title (str): (optional)
        - xlabel (str): (optional)
        - ylabel (str): (optional)
    
    Returns:
        - fig: plt figure object
        - ax: plt axis object
        - stats_results (dict): results from one_way_anova
    """
    
    stats_results = one_way_anova(data_groups)
    
    # Calculate means & SEM
    means = [np.mean(group) for group in data_groups]
    sems = [stats.sem(group) for group in data_groups]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data_groups))
    
    # Plot line with error bars
    ax.errorbar(x, means, yerr=sems, fmt='o-', capsize=5, linewidth=3, markersize=8)

    # Add plot elements
    if group_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=20)
    
    if title:
        ax.set_title(title, fontsize=30)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=24)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=24)
    
    ax.tick_params(axis='y', labelsize=20)
    
    # Add ANOVA results to plot
    anova_text = f"One-way ANOVA: \nF = {stats_results['f_stat']:.3f}\np = {stats_results['p_value']:.2e}"
    ax.text(0.95, 0.95, anova_text, transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax, stats_results

def perform_chi_square_test(success_counts, total_counts, trial_numbers):
    success_counts = np.array(success_counts)
    total_counts = np.array(total_counts)

    # overall chi square test
    contingency_table = np.array([
        [success_counts[i], total_counts[i] - success_counts[i]]
        for i in range(len(trial_numbers))
    ])
    
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    # pairwise chi square test
    n = len(trial_numbers)
    pairwise_p_values = []
    pair_labels = []
    
    for i in range(n):
        for j in range(i+1, n):
            # Create 2x2 contingency table for this pair
            pair_table = np.array([
                [success_counts[i], total_counts[i] - success_counts[i]],
                [success_counts[j], total_counts[j] - success_counts[j]]
            ])
            
            _, p_val = stats.fisher_exact(pair_table)
            pairwise_p_values.append(p_val)
            pair_labels.append(f"{trial_numbers[i]} vs {trial_numbers[j]}")
    
    adjusted_p_values = multipletests(pairwise_p_values, method='bonferroni')[1]
    
    return {
        "chi2": chi2,
        "p_value": p_value,
        "pairwise_tests": list(zip(pair_labels, pairwise_p_values, adjusted_p_values))
    }

def plot_chi_square_test(success_counts, total_counts, group_labels, title="", xlabel="", ylabel=""):
    stats_results = perform_chi_square_test(success_counts, total_counts, group_labels)
    
    print("Pairwise comparisons (with Bonferroni correction):")
    for pair, p_val, adj_p_val in stats_results["pairwise_tests"]:
        print(f"{pair}: p={p_val:.4f} (adjusted p={adj_p_val:.4f})")
    
    success_counts = np.array(success_counts)
    total_counts = np.array(total_counts)
    proportions = success_counts / total_counts
    
    #p = success_counts / total_counts if total_counts > 0 else 0
    sem = [np.sqrt((p * (1 - p)) / n) if n > 0 else 0 for p, n in zip(proportions, total_counts)]
    
    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(proportions))
    
    #sem = helper.get_sem(success_counts, total_counts)
    ax.bar(x, proportions, yerr=sem, capsize=5)
    
    #if stats_results['p_value'] < 0.05:
        # Add asterisk above the highest bar
        #max_height = max(proportions) + max(errors)
        #ax.text(len(proportions)/2, max_height * 1.1, '*', 
                #ha='center', va='bottom', fontsize=12)
    
    # Customize plot
    if group_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Add chi-square results to plot
    stats_text = (f'Chi-square test:\n'
                 f'χ² = {stats_results["chi2"]:.3f}\n'
                 f'p = {stats_results["p_value"]:.2e}')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax, stats_results
    
    