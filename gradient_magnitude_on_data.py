import numpy as np
import json
from scipy.spatial import KDTree


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


def compute_gradient_magnitude(X_samples, y_samples, k=12):
    tree = KDTree(X_samples)
    gradient_magnitudes = []

    for i, x in enumerate(X_samples):
        dists, indices = tree.query(x, k=k)
        f_neighbors = y_samples[indices]
        grad_magnitudes = np.abs((f_neighbors - y_samples[i]) / (dists + 1e-8))  # Avoid division by zero
        avg_grad_magnitude = np.mean(grad_magnitudes)
        gradient_magnitudes.append(avg_grad_magnitude)

    return np.array(gradient_magnitudes)


json_file = "schwefel_10D_ImprovementBased.json"

X_samples, y_samples = load_json_data(json_file)

gradient_magnitudes = compute_gradient_magnitude(X_samples, y_samples, k=12)

avg_gradient_magnitude = np.mean(gradient_magnitudes)
std_gradient_magnitude = np.std(gradient_magnitudes)

print(f"Average Gradient Magnitude: {avg_gradient_magnitude:.6f} ± {std_gradient_magnitude:.6f}")