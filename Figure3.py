import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Ensure the 'images' folder exists
if not os.path.exists('images'):
    os.makedirs('images')

# Define the free energy landscape with different states
def free_energy(x, state):
    if state == 'unmanifested':
        np.random.seed(0)  # For reproducibility
        components = [
            0.05 * np.sin(2 * np.pi * x / 4),
            0.02 * np.sin(4 * np.pi * x / 4),
            0.01 * np.sin(8 * np.pi * x / 4)
        ]
        np.random.shuffle(components)
        return sum(components)  # Wavy pattern with randomized order of superpositions
    elif state == 'manifested':
        return -0.5 * np.exp(-(x - 2)**2) - 0.2 * np.exp(-(x + 2)**2) - 0.15 * np.exp(-(x - 1)**2) - 0.15 * np.exp(-(x + 1)**2) 
    elif state == 'activated':
        return -0.7 * np.exp(-(x - 2)**2) - 0.3 * np.exp(-(x + 2)**2) - 0.3 * np.exp(-(x - 1)**2) - 0.3 * np.exp(-(x + 1)**2)
# Generate x values for the plot
x = np.linspace(-4, 4, 1000)

# Calculate free energy values for each state
y_unmanifested = free_energy(x, 'unmanifested')
y_manifested = free_energy(x, 'manifested')
y_activated = free_energy(x, 'activated')

# Find all minimum points (attractors) for each state
minima_indices_unmanifested, _ = find_peaks(-y_unmanifested)
minima_x_unmanifested = x[minima_indices_unmanifested]
minima_y_unmanifested = y_unmanifested[minima_indices_unmanifested]

minima_indices_manifested, _ = find_peaks(-y_manifested)
minima_x_manifested = x[minima_indices_manifested]
minima_y_manifested = y_manifested[minima_indices_manifested]

minima_indices_activated, _ = find_peaks(-y_activated)
minima_x_activated = x[minima_indices_activated]
minima_y_activated = y_activated[minima_indices_activated]

# Plot the free energy landscapes
plt.figure(figsize=(12, 8))
plt.plot(x, y_unmanifested, label='Unmanifested State')
plt.plot(x, y_manifested, label='Manifested State')
plt.plot(x, y_activated, label='Activated State')

# Identify global minima for each state
global_min_index_unmanifested = np.argmin(y_unmanifested)
global_min_index_manifested = np.argmin(y_manifested)
global_min_index_activated = np.argmin(y_activated)

# Highlight the global minima (core attractors) with dots
plt.scatter(x[global_min_index_manifested], y_manifested[global_min_index_manifested], color='red', marker='o', s=100, zorder=5, label='Core Attractor (Manifested)')
plt.scatter(x[global_min_index_activated], y_activated[global_min_index_activated], color='blue', marker='o', s=100, zorder=5, label='Core Attractor (Activated)')

# Highlight the local minima (subordinate attractors) with crosses
plt.scatter(minima_x_manifested, minima_y_manifested, color='red', marker='x', zorder=5, label='Subordinate Attractor (Manifested)')
plt.scatter(minima_x_activated, minima_y_activated, color='blue', marker='x', zorder=5, label='Subordinate Attractor (Activated)')

# Annotate the energy barriers for each state
def annotate_energy_barriers(x, y, minima_indices, global_min_index, color):
    for i, min_index in enumerate(minima_indices):
        if (min_index != global_min_index):  # Skip the core attractor
            # Find the saddle point between the core attractor and the subordinate attractor
            left_bound = min(global_min_index, min_index)
            right_bound = max(global_min_index, min_index)
            barrier_index = np.argmax(y[left_bound:right_bound]) + left_bound
            barrier_height = y[barrier_index]
            core_attractor_height = y[global_min_index]
            energy_barrier = barrier_height - core_attractor_height

            # Plot vertical dashed line for the energy barrier
            plt.plot([x[barrier_index], x[barrier_index]], [core_attractor_height, barrier_height], color=color, linestyle='--')

            # Annotate the energy barrier height
            plt.annotate('Energy Barrier\nManifested State', 
                        (x[barrier_index], (core_attractor_height + barrier_height) / 2 + .1),  # Adjusted position
                        textcoords="offset points", xytext=(0,10), ha='center', color=color)

# Annotate energy barriers for manifested state only
annotate_energy_barriers(x, y_manifested, minima_indices_manifested, global_min_index_manifested, 'black')

# Add a horizontal line for the zero energy level
plt.axhline(0, color='grey', linestyle='--', label='_nolegend_', alpha=0.5)

# Add a horizontal line for the core attractor of the manifested state
plt.axhline(y_manifested[global_min_index_manifested], color='grey', linestyle='--', label='_nolegend_', alpha=0.5)

# Annotate the binding energy for the manifested state
binding_energy_manifested = -y_manifested[global_min_index_manifested]
plt.plot([x[global_min_index_manifested], x[global_min_index_manifested]], [0, y_manifested[global_min_index_manifested]], color='black', linestyle='--')
plt.annotate('Binding Energy\nManifested State', 
             xy=(x[global_min_index_manifested], y_manifested[global_min_index_manifested]/2), 
             textcoords="offset points", xytext=(10,0), ha='center', color='black')

plt.xlabel('x')
plt.ylabel('Illustrative Free Energy', fontweight='bold')
plt.legend()
plt.title('Illustrative Free Energy Landscapes of a Neuronal Packet', fontweight='bold')

# Save the plot to the 'images' folder
plt.savefig('images/free_energy_landscape_with_barriers.png')

# Show the plot
plt.show()