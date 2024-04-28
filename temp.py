import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set the random seed for reproducibility
np.random.seed(0)

# Create a figure and a 3D axis
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# Number of electrodes
num_electrodes = 60

# Time vector from 0 to 2 seconds, sampled at 1000 Hz
time = np.linspace(0, 2, 2000)

# Simulate a spike wave across electrodes
for i in range(num_electrodes):
    # Base spike time, with added randomness
    base_spike_time = 0.5 + (i / float(num_electrodes)) * 0.1
    random_spike_time_offset = np.random.normal(0, 0.05)  # Noise in the spike timing
    spike_time = base_spike_time + random_spike_time_offset

    # Generate potentials within the desired range, adjusted amplitude
    potential = -45 * np.exp(-(time - spike_time)**2 / (2 * 0.01**2)) + 20

    # Introduce noise to the potential
    noise_level = 2
    potential_noise = potential + np.random.normal(0, noise_level, potential.shape)

    # Plot each electrode's potential over time
    ax.plot(time, [i]*len(time), potential_noise, color='gray', alpha = 0.3, linewidth = 0.5)

# Set labels
ax.set_xlabel('Time (s)')
ax.set_ylabel('Electrode (n)')
ax.set_zlabel('Potential (µV)')

# Set the limits of the plot with the z-axis limit from +20 to -70
ax.set_xlim([0, 2])
ax.set_ylim([0, num_electrodes])
ax.set_zlim([20, -70])  # Z-axis limits set from +20 to -70

# Manually set the z-axis tick labels to reflect the negative potential
ax.set_zticks([20, 0, -20, -40, -60, -70])
ax.set_zticklabels(['20', '0', '-20', '-40', '-60', '-70'])

# Set the viewing angle to show the 3D effect
ax.view_init(elev=40, azim=-70)

# Hide the grid and the panes
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Show the plot
plt.show()
