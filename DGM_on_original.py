
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def schwefel_nd(x):
    n = len(x)
    return 418.9828872724339 * n - sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)

def schwefel_nd_normalized(x):
    return schwefel_nd(x) / 10000.0

def schaffer_nd(x):
    return sum(
        ((xi**2 + xj**2) ** 0.25) * (np.sin(50 * (xi**2 + xj**2) ** 0.1) ** 2 + 1)
        for xi, xj in zip(x[:-1], x[1:])
    )

def schaffer_nd_normalized(x):
    return schaffer_nd(x) / 250.0

def discrete_gradient_method(start, X, y, k=12, learning_rate=10, max_iters=100, tolerance=0.00025):
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    x = start.copy()
    path = [x.copy()]
    idx = np.argmin(np.linalg.norm(X - x, axis=1))
    val = y[idx]
    vals = [val]

    for _ in range(max_iters):
        dists, indices = nn.kneighbors([x])
        neighbors = X[indices[0][1:]]
        f_neighbors = y[indices[0][1:]]

        dists = np.maximum(dists[0][1:], 1e-8)
        gradients = [(f - val) / (np.linalg.norm(xn - x)**2 + 1e-8) * (xn - x) for f, xn in zip(f_neighbors, neighbors)]
        weights = 1 / (dists + 1e-8)
        weights /= np.sum(weights)
        avg_grad = np.sum(weights[:, None] * gradients, axis=0)

        x -= learning_rate * avg_grad
        x = np.clip(x, X.min(axis=0), X.max(axis=0))

        new_val = y[np.argmin(np.linalg.norm(X - x, axis=1))]
        path.append(x.copy())
        vals.append(new_val)

        if np.linalg.norm(avg_grad) < tolerance:
            break

    return vals, len(path), x

def run_dgm_on_real_function(problem='schwefel', n_samples=100000, dim=10, num_runs=100):
    np.random.seed(42)
    if problem == 'schwefel':
        X = np.random.uniform(-500, 500, (n_samples, dim))
        y = np.array([schwefel_nd_normalized(x) for x in X])
        real_func = schwefel_nd
    elif problem == 'schaffer':
        X = np.random.uniform(-100, 100, (n_samples, dim))
        y = np.array([schaffer_nd_normalized(x) for x in X])
        real_func = schaffer_nd
    else:
        raise ValueError("Unknown problem")

    starts = X[np.random.choice(len(X), size=num_runs, replace=False)]
    results = [discrete_gradient_method(s, X, y) for s in starts]

    final_vals = [r[0][-1] for r in results]
    steps = [r[1] for r in results]
    real_vals = [real_func(r[2]) for r in results]

    print(f"Avg. final normalized value: {np.mean(final_vals):.4f}")
    print(f"Avg. real {problem} value: {np.mean(real_vals):.2f} ± {np.std(real_vals):.2f}")
    print(f"Avg. DGM steps: {np.mean(steps):.2f} ± {np.std(steps):.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(real_vals, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(f"Real {problem.capitalize()} Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(steps, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Steps to Convergence")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# run_dgm_on_real_function('schwefel')
run_dgm_on_real_function('schaffer')