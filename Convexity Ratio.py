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
from scipy.spatial import KDTree


def schwefel(x):
    return ((418.9828872724339 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))))/10000


def schaffer(x):
    sum_terms = 0.0
    for i in range(len(x) - 1):
        xi2_xj2 = x[i]**2 + x[i + 1]**2
        term1 = xi2_xj2 ** 0.25
        term2 = np.sin(50 * xi2_xj2 ** 0.10) ** 2
        sum_terms += term1 * (term2 + 1.0)
    return sum_terms/250

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

# Generate random points
# num_points = 10000
# X_samples = np.random.uniform(-100, 100, (num_points, 2))
# y_samples = np.array([schaffer(x[0], x[1]) for x in X_samples])

json_file = "schaffer_10D_FixedTarget.json"
X_samples, y_samples = load_json_data(json_file)



def compute_convexity_ratio(X_samples, y_samples, num_pairs=1000, num_interpolations=10):

    n = len(X_samples)
    cr_counter = 0
    total = 0
    tree = KDTree(X_samples)
    y_samples = np.array(y_samples)

    for _ in range(num_pairs):
        i, j = np.random.choice(n, size=2, replace=False)
        xa, xb = X_samples[i], X_samples[j]
        fa, fb = y_samples[i], y_samples[j]

        for l in range(1, num_interpolations + 1):
            lam = l / (num_interpolations + 1)
            x_lambda = (1 - lam) * xa + lam * xb
            f_interp = (1 - lam) * fa + lam * fb

            _, idx = tree.query(x_lambda)
            f_near = y_samples[idx]

            if f_near <= f_interp:
                cr_counter += 1
            total += 1

    convexity_ratio = cr_counter / total if total > 0 else 0
    return convexity_ratio


convexity_ratio = compute_convexity_ratio(X_samples, y_samples)
print(f"Convexity Ratio: {convexity_ratio:.6f}")