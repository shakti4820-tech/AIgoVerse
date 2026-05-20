import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles, load_iris, load_wine

DATASETS = {
    "Moons (2D)": "moons",
    "Circles (2D)": "circles",
    "Blobs (2D)": "blobs",
    "Iris (4D)": "iris",
    "Wine (13D)": "wine",
}


def load_dataset(name, n_samples=300, noise=0.2):
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        feature_names = ["Feature 1", "Feature 2", "Feature 3"]
        third_feature = X[:, 0] * 0.7 + X[:, 1] * 0.5 + np.random.normal(0, noise * 0.15, len(X))
        X = np.column_stack([X, third_feature])
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        feature_names = ["Feature 1", "Feature 2", "Feature 3"]
        third_feature = X[:, 0] * 0.5 - X[:, 1] * 0.3 + np.random.normal(0, noise * 0.15, len(X))
        X = np.column_stack([X, third_feature])
    elif name == "blobs":
        X, y = make_classification(n_samples=n_samples, n_features=2,
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
        feature_names = ["Feature 1", "Feature 2", "Feature 3"]
        third_feature = X[:, 0] * 0.6 + X[:, 1] * 0.4 + np.random.normal(0, noise * 0.15, len(X))
        X = np.column_stack([X, third_feature])
    elif name == "iris":
        d = load_iris()
        X, y = d.data[:, :3], d.target
        feature_names = [d.feature_names[i] for i in range(3)]
        
        # Resample to match n_samples for architectural consistency
        np.random.seed(42)
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        X = X[indices]
        y = y[indices]
        
        # Add noise proportional to the scale of each feature
        if noise > 0:
            stds = np.std(X, axis=0)
            stds[stds == 0] = 1.0
            X = X + np.random.normal(0, noise * 0.2 * stds, X.shape)
            
    elif name == "wine":
        d = load_wine()
        X, y = d.data[:, :3], d.target
        feature_names = [d.feature_names[i] for i in range(3)]
        
        # Resample to match n_samples for architectural consistency
        np.random.seed(42)
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        X = X[indices]
        y = y[indices]
        
        # Add noise proportional to the scale of each feature
        if noise > 0:
            stds = np.std(X, axis=0)
            stds[stds == 0] = 1.0
            X = X + np.random.normal(0, noise * 0.2 * stds, X.shape)
    return X, y, feature_names
