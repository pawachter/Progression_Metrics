"""
Distance Metrics for Domain Adaptation
Implements various statistical distance measures between distributions
OPTIMIZED: Vectorized operations, reduced scipy dependencies, caching
"""

import numpy as np
import tensorflow as tf
from functools import lru_cache


def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples - OPTIMIZED
    
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
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    
    if kernel == 'rbf':
        # RBF (Gaussian) kernel - optimized vectorized version
        XX = rbf_kernel_fast(X, X, gamma)
        YY = rbf_kernel_fast(Y, Y, gamma)
        XY = rbf_kernel_fast(X, Y, gamma)
    elif kernel == 'linear':
        # Linear kernel (dot product) - use einsum for better performance
        XX = np.einsum('ij,kj->ik', X, X)
        YY = np.einsum('ij,kj->ik', Y, Y)
        XY = np.einsum('ij,kj->ik', X, Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    mmd_squared = XX.mean() - 2 * XY.mean() + YY.mean()
    
    # Return MMD (take square root, but ensure non-negative due to numerical issues)
    return float(np.sqrt(max(0, mmd_squared)))


def rbf_kernel_fast(X, Y, gamma=1.0):
    """
    Compute RBF (Gaussian) kernel matrix - OPTIMIZED VECTORIZED VERSION
    
    K(x, y) = exp(-gamma * ||x - y||^2)
    
    Uses optimized vectorized computation without scipy dependency.
    Exploits ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
    
    Args:
        X: First set of samples (n_samples_x, n_features)
        Y: Second set of samples (n_samples_y, n_features)
        gamma: Kernel bandwidth parameter
    
    Returns:
        Kernel matrix (n_samples_x, n_samples_y)
    """
    # Compute squared norms efficiently
    X_sqnorms = np.einsum('ij,ij->i', X, X)[:, np.newaxis]  # (n, 1)
    Y_sqnorms = np.einsum('ij,ij->i', Y, Y)[np.newaxis, :]  # (1, m)
    
    # Compute dot product
    XY = np.dot(X, Y.T)  # (n, m)
    
    # Compute squared distances using the identity
    pairwise_sq_dists = X_sqnorms + Y_sqnorms - 2 * XY
    
    # Ensure non-negative (numerical stability)
    pairwise_sq_dists = np.maximum(pairwise_sq_dists, 0)
    
    # Apply RBF kernel
    return np.exp(-gamma * pairwise_sq_dists)


def rbf_kernel(X, Y, gamma=1.0):
    """Legacy wrapper - redirects to optimized version"""
    return rbf_kernel_fast(X, Y, gamma)



def compute_kl_divergence(X, Y, n_bins=50, epsilon=1e-10):
    """
    Compute KL Divergence between two distributions - OPTIMIZED
    
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
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    
    # Flatten if 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_features = X.shape[1]
    
    # Vectorized approach: compute all histograms at once when possible
    if n_features > 1:
        kl_divs = np.zeros(n_features, dtype=np.float32)
        
        # Process in chunks if too large
        for feature_idx in range(n_features):
            x_feature = X[:, feature_idx]
            y_feature = Y[:, feature_idx]
            
            # Determine common bin edges
            min_val = min(x_feature.min(), y_feature.min())
            max_val = max(x_feature.max(), y_feature.max())
            
            if np.isclose(min_val, max_val):
                kl_divs[feature_idx] = 0.0
                continue
            
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            # Compute histograms (normalized to form probability distributions)
            p, _ = np.histogram(x_feature, bins=bins)
            q, _ = np.histogram(y_feature, bins=bins)
            
            # Normalize to probabilities and add epsilon
            p = (p + epsilon) / (p.sum() + n_bins * epsilon)
            q = (q + epsilon) / (q.sum() + n_bins * epsilon)
            
            # Compute KL divergence manually (faster than scipy for single computation)
            kl_divs[feature_idx] = np.sum(p * np.log(p / q))
        
        return float(np.mean(kl_divs))
    else:
        # Single feature case
        x_feature = X.ravel()
        y_feature = Y.ravel()
        
        min_val = min(x_feature.min(), y_feature.min())
        max_val = max(x_feature.max(), y_feature.max())
        
        if np.isclose(min_val, max_val):
            return 0.0
        
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        p, _ = np.histogram(x_feature, bins=bins)
        q, _ = np.histogram(y_feature, bins=bins)
        
        # Normalize and add epsilon
        p = (p + epsilon) / (p.sum() + n_bins * epsilon)
        q = (q + epsilon) / (q.sum() + n_bins * epsilon)
        
        return float(np.sum(p * np.log(p / q)))


def compute_wasserstein_distance(X, Y):
    """
    Compute 1-Wasserstein distance (Earth Mover's Distance) between two distributions - OPTIMIZED
    
    For multi-dimensional data, computes Wasserstein distance per feature and averages.
    Uses optimized sorting-based method for 1D Wasserstein distance.
    
    Args:
        X: Source domain samples (n_samples_x, n_features)
        Y: Target domain samples (n_samples_y, n_features)
    
    Returns:
        Average Wasserstein distance across features (float)
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    
    # Flatten if 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_features = X.shape[1]
    
    # Vectorized Wasserstein computation for 1D distributions
    # W_1(P, Q) = integral |F_P^{-1}(t) - F_Q^{-1}(t)| dt
    # For discrete distributions: sort both and take L1 norm
    wasserstein_dists = np.zeros(n_features, dtype=np.float32)
    
    for feature_idx in range(n_features):
        x_sorted = np.sort(X[:, feature_idx])
        y_sorted = np.sort(Y[:, feature_idx])
        
        # Interpolate to common support
        n_x, n_y = len(x_sorted), len(y_sorted)
        
        # Fast computation using sorting
        if n_x == n_y:
            wasserstein_dists[feature_idx] = np.mean(np.abs(x_sorted - y_sorted))
        else:
            # Use linspace for equal-weight samples
            all_values = np.concatenate([x_sorted, y_sorted])
            all_values.sort()
            
            # Cumulative distributions
            cdf_x = np.searchsorted(x_sorted, all_values, side='right') / n_x
            cdf_y = np.searchsorted(y_sorted, all_values, side='right') / n_y
            
            wasserstein_dists[feature_idx] = np.trapz(np.abs(cdf_x - cdf_y), all_values)
    
    return float(np.mean(wasserstein_dists))


def compute_entropy(logits, epsilon=1e-10):
    """
    Compute average entropy of predictions - OPTIMIZED
    
    Entropy = -sum(p * log(p)) for predicted probabilities
    High entropy indicates uncertain predictions
    
    Args:
        logits: Model predictions (n_samples, n_classes)
                Can be logits or probabilities
        epsilon: Small constant to avoid log(0)
    
    Returns:
        Average entropy across samples (float)
    """
    logits = np.asarray(logits, dtype=np.float32)
    
    if len(logits) == 0:
        return 0.0
    
    # Convert logits to probabilities using softmax if needed
    # Check if already probabilities (sum to ~1)
    row_sums = np.sum(logits, axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.1):
        # Apply softmax with numerical stability
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    else:
        probs = logits
    
    # Clip and compute entropy in one vectorized operation
    probs = np.clip(probs, epsilon, 1.0)
    
    # Compute entropy for all samples at once (vectorized)
    entropies = -np.sum(probs * np.log(probs), axis=1)
    
    # Return average entropy
    return float(np.mean(entropies))

