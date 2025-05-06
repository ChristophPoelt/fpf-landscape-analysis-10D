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

def schwefel(x):
    return 418.9828872724339 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def h1(x1, x2):
    term1 = np.sin(x1 - x2 / 8) ** 2
    term2 = np.sin(x2 + x1 / 8) ** 2
    denominator = np.sqrt((x1 - 8.6998) ** 2 + (x2 - 6.7665) ** 2) + 1
    return -((term1 + term2) / denominator) + 2

def schaffer(x):
    sum_terms = 0.0
    for i in range(len(x) - 1):
        xi2_xj2 = x[i]**2 + x[i + 1]**2
        term1 = xi2_xj2 ** 0.25
        term2 = np.sin(50 * xi2_xj2 ** 0.10) ** 2
        sum_terms += term1 * (term2 + 1.0)
    return sum_terms

    with open(json_file, 'r') as f:
        data = json.load(f)

    X_samples = []
    y_samples = []

    for entry in data:
        for individual in entry['individualsWithFPF']:
            X_samples.append([individual['x1'], individual['x2']])
            y_samples.append(individual['fpfValue'])

    return np.array(X_samples), np.array(y_samples)


def discrete_gradient_method(starting_point, X_samples, y_samples, k=12, learning_rate=10, max_iters=100, tolerance=1e-12):
    tree = KDTree(X_samples)
    x = np.array(starting_point, dtype=np.float64)
    path = [x.copy()]
    function_values = [y_samples[np.argmin(np.linalg.norm(X_samples - x, axis=1))]]

    for step in range(max_iters):
        dists, indices = tree.query(x, k=k)
        neighbors = X_samples[indices]
        f_neighbors = y_samples[indices]

        # Vermeidung von Division durch Null
        dists = np.maximum(dists, 1e-8)

        gradients = np.array([(f_neighbors[i] - function_values[-1]) / (np.linalg.norm(neighbors[i] - x) ** 2 + 1e-8) * (neighbors[i] - x)
                              for i in range(k)])

        weights = 1 / (dists + 1e-8)
        weights /= np.sum(weights)
        avg_gradient = np.sum(weights[:, None] * gradients, axis=0)

        x -= learning_rate * avg_gradient
        x = np.clip(x, np.min(X_samples, axis=0), np.max(X_samples, axis=0))

        if not np.all(np.isfinite(x)):
            break

        path.append(x.copy())
        function_values.append(y_samples[np.argmin(np.linalg.norm(X_samples - x, axis=1))])

        if np.linalg.norm(avg_gradient) < tolerance:
            break

    # Berechnung des realen Schwefel-Wertes (oder andere Probleminstanzen) am Endpunkt
    final_schwefel_value = schaffer(x)
    # final_schwefel_value = schwefel(x)

    return function_values, step + 1, final_schwefel_value

def main(json_file, num_runs=100):
    X_samples, y_samples = load_json_data(json_file)
    start_points = X_samples[np.random.choice(len(X_samples), num_runs, replace=False)]

    results = [discrete_gradient_method(start, X_samples, y_samples) for start in start_points]

    final_values = np.array([res[0][-1] for res in results])
    final_schwefel_values = np.array([res[2] for res in results])
    num_steps = np.array([res[1] for res in results])

    avg_final_value = np.mean(final_values)
    avg_final_schwefel = np.mean(final_schwefel_values)
    std_final_schwefel = np.std(final_schwefel_values)
    avg_steps = np.mean(num_steps)
    std_steps = np.std(num_steps)

    print(f"Durchschnittlicher finaler FPF-Wert: {avg_final_value:.2f}")
    print(f"Durchschnittlicher realer Schwefel-Wert: {avg_final_schwefel:.2f} ± {std_final_schwefel:.2f}")
    print(f"Durchschnittliche Anzahl an Schritten bis zur Konvergenz: {avg_steps:.2f} ± {std_steps:.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(final_schwefel_values, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("DGM results' H1 target value")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(num_steps, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Number of steps to convergence")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


main("schaffer_10D_FixedTarget.json")