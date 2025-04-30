def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    X_samples = []
    y_samples = []
    for entry in data:
        for individual in entry['individualsWithFPF']:
            if individual['fpfValue'] != 1.0:
                X_samples.append(individual['genome'])
                y_samples.append(individual['fpfValue'])
    return np.array(X_samples), np.array(y_samples)

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


# Load data from JSON file
# OLD load_json_data removed
# def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    X_samples = []
    y_samples = []

    for entry in data:
        for individual in entry['individualsWithFPF']:
            if individual['fpfValue'] != 1.0:  # Exclude fpfValue of 1.0
                X_samples.append([individual['x1'], individual['x2']])
                y_samples.append(individual['fpfValue'])

    return np.array(X_samples), np.array(y_samples)


# Function to compute the gradient magnitude
def compute_gradient_magnitude(X_samples, y_samples, k=12):
    tree = KDTree(X_samples)  # KD-Tree for fast k-NN search
    gradient_magnitudes = []

    for i, x in enumerate(X_samples):
        # Find k-nearest neighbors
        dists, indices = tree.query(x, k=k)
        neighbors = X_samples[indices]
        f_neighbors = y_samples[indices]

        # Compute gradient magnitudes for each neighbor
        grad_magnitudes = np.abs((f_neighbors - y_samples[i]) / (dists + 1e-8))  # Avoid division by zero

        # Compute average gradient magnitude
        avg_grad_magnitude = np.mean(grad_magnitudes)
        gradient_magnitudes.append(avg_grad_magnitude)

    return np.array(gradient_magnitudes)


# Load your data
json_file = "_hpi_schwefelFixedTarget.json"
X_samples, y_samples = load_json_data(json_file)

# Compute gradient magnitudes
gradient_magnitudes = compute_gradient_magnitude(X_samples, y_samples, k=12)

# Compute average and standard deviation of gradient magnitudes
avg_gradient_magnitude = np.mean(gradient_magnitudes)
std_gradient_magnitude = np.std(gradient_magnitudes)

# Print results
print(f"Average Gradient Magnitude: {avg_gradient_magnitude:.6f} ± {std_gradient_magnitude:.6f}")

# Create a heatmap of gradient magnitudes
plt.figure(figsize=(8, 6))
sc = # Removed for 10D(X_samples[:, 0], X_samples[:, 1], c=gradient_magnitudes, cmap='viridis', s=5, alpha=0.7)
plt.colorbar(sc, label="Gradient Magnitude")
# Removed for 10D
# Removed for 10D
# Removed for 10D("Heatmap of Gradient Magnitude (Landscape Smoothness)")
plt.show()