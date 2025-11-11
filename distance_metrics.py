"""
Distance Metrics for Domain Adaptation
Implements various statistical distance measures between distributions
"""

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from scipy.stats import entropy as scipy_entropy


def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples
    
    MMD is a kernel-based statistical test that measures the distance
    between two probability distributions.
    
    Args:
        X: Source domain samples (n_samples_x, n_features)
        Y: Target domain samples (n_samples_y, n_features)
        kernel: Kernel type ('rbf' or 'linear')
        gamma: Kernel bandwidth parameter for RBF kernel
    
    Returns:
        MMD distance (float)
    """
    X = np.array(X)
    Y = np.array(Y)
    
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    
    if kernel == 'rbf':
        # RBF (Gaussian) kernel
        XX = rbf_kernel(X, X, gamma)
        YY = rbf_kernel(Y, Y, gamma)
        XY = rbf_kernel(X, Y, gamma)
    elif kernel == 'linear':
        # Linear kernel (dot product)
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    mmd_squared = XX.mean() - 2 * XY.mean() + YY.mean()
    
    # Return MMD (take square root, but ensure non-negative due to numerical issues)
    return np.sqrt(max(0, mmd_squared))


def rbf_kernel(X, Y, gamma=1.0):
    """
    Compute RBF (Gaussian) kernel matrix
    
    K(x, y) = exp(-gamma * ||x - y||^2)
    
    Args:
        X: First set of samples (n_samples_x, n_features)
        Y: Second set of samples (n_samples_y, n_features)
        gamma: Kernel bandwidth parameter
    
    Returns:
        Kernel matrix (n_samples_x, n_samples_y)
    """
    # Compute pairwise squared Euclidean distances
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    
    # Apply RBF kernel
    return np.exp(-gamma * pairwise_sq_dists)


def compute_kl_divergence(X, Y, n_bins=50, epsilon=1e-10):
    """
    Compute KL Divergence between two distributions
    
    KL(P||Q) = sum(P(x) * log(P(x) / Q(x)))
    
    Approximates distributions using histograms for high-dimensional data.
    For multi-dimensional data, we compute KL divergence per feature and average.
    
    Args:
        X: Source domain samples (n_samples_x, n_features)
        Y: Target domain samples (n_samples_y, n_features)
        n_bins: Number of bins for histogram approximation
        epsilon: Small constant to avoid log(0)
    
    Returns:
        Average KL divergence across features (float)
    """
    X = np.array(X)
    Y = np.array(Y)
    
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    
    # Flatten if 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_features = X.shape[1]
    kl_divs = []
    
    for feature_idx in range(n_features):
        x_feature = X[:, feature_idx]
        y_feature = Y[:, feature_idx]
        
        # Determine common bin edges
        min_val = min(x_feature.min(), y_feature.min())
        max_val = max(x_feature.max(), y_feature.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Compute histograms (normalized to form probability distributions)
        p, _ = np.histogram(x_feature, bins=bins)
        q, _ = np.histogram(y_feature, bins=bins)
        
        # Normalize to probabilities
        p = p / (p.sum() + epsilon)
        q = q / (q.sum() + epsilon)
        
        # Add epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon
        
        # Compute KL divergence using scipy
        kl_div = scipy_entropy(p, q)
        kl_divs.append(kl_div)
    
    # Return average KL divergence across features
    return np.mean(kl_divs)


def compute_wasserstein_distance(X, Y):
    """
    Compute 1-Wasserstein distance (Earth Mover's Distance) between two distributions
    
    For multi-dimensional data, computes Wasserstein distance per feature and averages.
    Uses the scipy implementation which solves the optimal transport problem.
    
    Args:
        X: Source domain samples (n_samples_x, n_features)
        Y: Target domain samples (n_samples_y, n_features)
    
    Returns:
        Average Wasserstein distance across features (float)
    """
    from scipy.stats import wasserstein_distance
    
    X = np.array(X)
    Y = np.array(Y)
    
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    
    # Flatten if 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_features = X.shape[1]
    wasserstein_dists = []
    
    for feature_idx in range(n_features):
        x_feature = X[:, feature_idx]
        y_feature = Y[:, feature_idx]
        
        # Compute Wasserstein distance for this feature
        dist = wasserstein_distance(x_feature, y_feature)
        wasserstein_dists.append(dist)
    
    # Return average Wasserstein distance across features
    return np.mean(wasserstein_dists)


def compute_entropy(logits, epsilon=1e-10):
    """
    Compute average entropy of predictions
    
    Entropy = -sum(p * log(p)) for predicted probabilities
    High entropy indicates uncertain predictions
    
    Args:
        logits: Model predictions (n_samples, n_classes)
                Can be logits or probabilities
        epsilon: Small constant to avoid log(0)
    
    Returns:
        Average entropy across samples (float)
    """
    logits = np.array(logits)
    
    if len(logits) == 0:
        return 0.0
    
    # Convert logits to probabilities using softmax if needed
    # Check if already probabilities (sum to ~1)
    row_sums = np.sum(logits, axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.1):
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    else:
        probs = logits
    
    # Add epsilon to avoid log(0)
    probs = np.clip(probs, epsilon, 1.0)
    
    # Compute entropy for each sample
    entropies = -np.sum(probs * np.log(probs), axis=1)
    
    # Return average entropy
    return np.mean(entropies)
