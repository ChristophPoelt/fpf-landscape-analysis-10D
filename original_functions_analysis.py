import numpy as np
import json
from scipy.stats import entropy, pearsonr
from sklearn.neighbors import NearestNeighbors


def schaffer_nd(x):
    sum_terms = 0.0
    for i in range(len(x) - 1):
        xi2_xj2 = x[i]**2 + x[i + 1]**2
        term1 = xi2_xj2 ** 0.25
        term2 = np.sin(50 * xi2_xj2 ** 0.10) ** 2
        sum_terms += term1 * (term2 + 1.0)
    return sum_terms

def schaffer_nd_normalized(x):
    return schaffer_nd(x) / 250.0

def schwefel_nd(x):
    n = len(x)
    sum_term = sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])
    return 418.9828872724339 * n - sum_term

def schwefel_nd_normalized(x):
    return schwefel_nd(x) / 10000.0


def compute_convexity_ratio(y):
    convex_count = 0
    for i in range(1, len(y) - 1):
        if y[i] <= (y[i - 1] + y[i + 1]) / 2:
            convex_count += 1
    return convex_count / (len(y) - 2)

def compute_entropy(y, bins=30):
    hist, _ = np.histogram(y, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(entropy(hist, base=2))

def compute_fdc(X, y, optimum):
    distances = np.linalg.norm(X - optimum, axis=1)
    corr, _ = pearsonr(distances, y)
    return float(corr)

def compute_gradient_magnitude(X, y, k=12):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    gradients = []
    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]
        local_grads = []
        for j in range(1, k+1):
            x_j = X[indices[i][j]]
            y_j = y[indices[i][j]]
            delta_x = x_j - x_i
            norm = np.linalg.norm(delta_x)
            if norm > 0:
                grad = abs(y_j - y_i) / norm
                local_grads.append(grad)
        if local_grads:
            gradients.append(np.mean(local_grads))
    return float(np.mean(gradients)), float(np.std(gradients))


def run_full_analysis(problem='schwefel', n_points=100000, dim=10, seed=42):
    np.random.seed(seed)
    if problem == 'schwefel':
        X = np.random.uniform(-500, 500, (n_points, dim))
        y = np.array([schwefel_nd_normalized(x) for x in X])
        optimum = np.array([420.96874636] * dim)
    elif problem == 'schaffer':
        X = np.random.uniform(-100, 100, (n_points, dim))
        y = np.array([schaffer_nd_normalized(x) for x in X])
        optimum = np.zeros(dim)
    else:
        raise ValueError("Unknown problem")

    return {
        "convexity_ratio": compute_convexity_ratio(y),
        "entropy": compute_entropy(y),
        "fitness_distance_correlation": compute_fdc(X, y, optimum),
        "gradient_magnitude": {
            "mean": compute_gradient_magnitude(X, y)[0],
            "std": compute_gradient_magnitude(X, y)[1]
        }
    }

if __name__ == "__main__":
    result = {
        "schwefel": run_full_analysis("schwefel"),
        "schaffer": run_full_analysis("schaffer")
    }

    with open("function_analysis_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Analysis complete. Results saved to function_analysis_metrics.json")
