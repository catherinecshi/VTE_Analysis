import numpy as np
import matplotlib.pyplot as plt

def plot_zIdPhi(zIdPhi_values, save=None):
    """
    plot the mean and std of zIdPhi values

    Args:
        zIdPhi_values (dict):{choice: zIdPhis}
        save (str, optional): file path if saving is desired. Defaults to None.
    """
    
    # Collect all zIdPhi values from all trial types
    all_zIdPhis = []
    for zIdPhis in zIdPhi_values.values():
        all_zIdPhis.extend(zIdPhis)
    
    # Convert to a NumPy array for statistical calculations
    all_zIdPhis = np.array(all_zIdPhis)
    
    # Create a single plot
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(all_zIdPhis, bins=30, alpha=0.7, label="All Trial Types")
    
    # Calculate and plot the mean and standard deviation lines
    mean = float(np.mean(all_zIdPhis))
    std = float(np.std(all_zIdPhis))
    
    plt.axvline(mean, color="red", linestyle="dashed", linewidth=2, label="Mean")
    plt.axvline(mean + std, color="green", linestyle="dashed", linewidth=2, label="+1 STD")
    plt.axvline(mean - std, color="green", linestyle="dashed", linewidth=2, label="-1 STD")
    
    # Set the title and labels
    plt.title("Combined IdPhi Distribution Across All Trial Types")
    plt.xlabel("zIdPhi")
    plt.ylabel("Frequency")
    
    # Show the legend
    plt.legend()
    plt.tight_layout()
    
    if save:
        save_path = f"{save}/zIdPhi_Distribution.jpg"
        plt.savefig(save_path)
    else:
        plt.show()

