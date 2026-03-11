"""
Numeric Safety Utilities for EvoMind

This module provides utilities to prevent numerical instability (NaN/Inf)
from entering genomes and collapsing evolution.

CRITICAL: Always use these functions after:
- Mutation operations
- Crossover operations  
- Plasticity updates
- Weight normalization
- Any division operation
"""

import numpy as np
import torch
from typing import Union, Optional, Tuple, Any


def sanitize_array(
    arr: Union[np.ndarray, torch.Tensor],
    nan_replace: float = 0.0,
    posinf_replace: float = 1.0,
    neginf_replace: float = -1.0,
    copy: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Replace NaN and Inf values with safe finite values.
    
    Args:
        arr: Input array (numpy or torch)
        nan_replace: Value to replace NaN with
        posinf_replace: Value to replace +Inf with
        neginf_replace: Value to replace -Inf with
        copy: If True, return a copy; if False, modify in-place (numpy only)
    
    Returns:
        Sanitized array with all finite values
    """
    if isinstance(arr, np.ndarray):
        if copy:
            arr = arr.copy()
        # Use numpy's nan_to_num for efficiency
        return np.nan_to_num(
            arr,
            nan=nan_replace,
            posinf=posinf_replace,
            neginf=neginf_replace
        )
    elif isinstance(arr, torch.Tensor):
        # For torch tensors
        with torch.no_grad():
            # Replace NaN
            arr = torch.where(torch.isnan(arr), torch.tensor(nan_replace, dtype=arr.dtype, device=arr.device), arr)
            # Replace +Inf
            arr = torch.where(torch.isposinf(arr), torch.tensor(posinf_replace, dtype=arr.dtype, device=arr.device), arr)
            # Replace -Inf
            arr = torch.where(torch.isneginf(arr), torch.tensor(neginf_replace, dtype=arr.dtype, device=arr.device), arr)
        return arr
    else:
        # Fallback for other array-like types
        try:
            return np.nan_to_num(np.array(arr), nan=nan_replace, posinf=posinf_replace, neginf=neginf_replace)
        except Exception:
            return arr


def safe_divide(
    numerator: Union[np.ndarray, torch.Tensor, float],
    denominator: Union[np.ndarray, torch.Tensor, float],
    epsilon: float = 1e-8,
    default_value: float = 0.0
) -> Union[np.ndarray, torch.Tensor, float]:
    """
    Safe division that prevents division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value (protected by epsilon)
        epsilon: Small value added to denominator
        default_value: Value to return if result would be NaN/Inf
    
    Returns:
        Safe division result
    """
    # Handle numpy arrays
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        numerator = np.asarray(numerator)
        denominator = np.asarray(denominator)
        result = numerator / (denominator + epsilon)
        return sanitize_array(result, nan_replace=default_value, posinf_replace=default_value, neginf_replace=default_value)
    
    # Handle torch tensors
    if isinstance(numerator, torch.Tensor) or isinstance(denominator, torch.Tensor):
        if not isinstance(numerator, torch.Tensor):
            numerator = torch.tensor(numerator)
        if not isinstance(denominator, torch.Tensor):
            denominator = torch.tensor(denominator)
        result = numerator / (denominator + epsilon)
        # Check for non-finite values
        if not torch.isfinite(result).all():
            result = sanitize_array(result, nan_replace=default_value, posinf_replace=default_value, neginf_replace=default_value)
        return result
    
    # Handle scalar values
    try:
        result = numerator / (denominator + epsilon)
        if not np.isfinite(result):
            return default_value
        return result
    except Exception:
        return default_value


def safe_normalize(
    arr: Union[np.ndarray, torch.Tensor],
    axis: Optional[int] = None,
    epsilon: float = 1e-8,
    ord: Optional[int] = 2
) -> Union[np.ndarray, torch.Tensor]:
    """
    Safe normalization that prevents division by zero norm.
    
    Args:
        arr: Input array to normalize
        axis: Axis along which to normalize (None = global)
        epsilon: Small value added to norm
        ord: Order of the norm (2 = L2 norm, 1 = L1 norm, etc.)
    
    Returns:
        Normalized array
    """
    if isinstance(arr, np.ndarray):
        # Compute norm
        if ord == 2 or ord is None:
            norm = np.linalg.norm(arr, axis=axis, keepdims=True)
        elif ord == 1:
            norm = np.sum(np.abs(arr), axis=axis, keepdims=True)
        elif ord == np.inf:
            norm = np.max(np.abs(arr), axis=axis, keepdims=True)
        else:
            norm = np.linalg.norm(arr, ord=ord, axis=axis, keepdims=True)
        
        # Normalize with epsilon protection
        result = arr / (norm + epsilon)
        return sanitize_array(result)
    
    elif isinstance(arr, torch.Tensor):
        # Compute norm
        if ord == 2 or ord is None:
            norm = torch.norm(arr, dim=axis, keepdim=True)
        elif ord == 1:
            norm = torch.sum(torch.abs(arr), dim=axis, keepdim=True)
        elif ord == np.inf:
            norm = torch.max(torch.abs(arr), dim=axis, keepdim=True)[0]
        else:
            # Fallback to numpy for other norms
            norm = torch.tensor(np.linalg.norm(arr.cpu().numpy(), ord=ord, axis=axis, keepdims=True), 
                              dtype=arr.dtype, device=arr.device)
        
        # Normalize with epsilon protection
        result = arr / (norm + epsilon)
        return sanitize_array(result)
    
    else:
        raise TypeError(f"Unsupported array type: {type(arr)}")


def check_finite(
    arr: Union[np.ndarray, torch.Tensor, Any],
    raise_error: bool = False,
    context: str = ""
) -> bool:
    """
    Check if array contains only finite values (no NaN or Inf).
    
    Args:
        arr: Input array to check
        raise_error: If True, raise ValueError on non-finite values
        context: Context string for error messages
    
    Returns:
        True if all values are finite, False otherwise
    """
    is_finite = True
    
    if isinstance(arr, np.ndarray):
        is_finite = bool(np.isfinite(arr).all())
    elif isinstance(arr, torch.Tensor):
        is_finite = bool(torch.isfinite(arr).all().item())
    elif isinstance(arr, (int, float)):
        is_finite = bool(np.isfinite(arr))
    elif arr is None:
        is_finite = True  # None is considered "finite" (no data)
    else:
        # Try to convert to numpy
        try:
            is_finite = bool(np.isfinite(np.array(arr)).all())
        except Exception:
            # If we can't check, assume it's finite
            is_finite = True
    
    if not is_finite and raise_error:

        raise ValueError(f"Non-finite values detected{f' in {context}' if context else ''}")
    
    return is_finite


def safe_mean(
    arr: Union[np.ndarray, torch.Tensor],
    axis: Optional[int] = None,
    epsilon: float = 1e-8,
    default_value: float = 0.0
) -> Union[np.ndarray, torch.Tensor, float]:
    """
    Safe mean calculation that handles empty arrays and NaN.
    
    Args:
        arr: Input array
        axis: Axis along which to compute mean
        epsilon: Not used (for API consistency)
        default_value: Default value if computation fails
    
    Returns:
        Mean value or default_value if computation fails
    """
    try:
        if isinstance(arr, np.ndarray):
            if arr.size == 0:
                return default_value
            result = np.mean(arr, axis=axis)
            return sanitize_array(result, nan_replace=default_value, posinf_replace=default_value, neginf_replace=default_value)
        elif isinstance(arr, torch.Tensor):
            if arr.numel() == 0:
                return torch.tensor(default_value, dtype=arr.dtype, device=arr.device)
            result = torch.mean(arr, dim=axis)
            return sanitize_array(result, nan_replace=default_value, posinf_replace=default_value, neginf_replace=default_value)
        else:
            return default_value
    except Exception:
        return default_value


def safe_std(
    arr: Union[np.ndarray, torch.Tensor],
    axis: Optional[int] = None,
    epsilon: float = 1e-8,
    default_value: float = 1.0
) -> Union[np.ndarray, torch.Tensor, float]:
    """
    Safe standard deviation calculation with epsilon protection.
    
    Args:
        arr: Input array
        axis: Axis along which to compute std
        epsilon: Added to result to prevent zero std
        default_value: Default value if computation fails
    
    Returns:
        Std value (always >= epsilon) or default_value if computation fails
    """
    try:
        if isinstance(arr, np.ndarray):
            if arr.size == 0:
                return default_value
            result = np.std(arr, axis=axis)
            result = sanitize_array(result, nan_replace=default_value, posinf_replace=default_value, neginf_replace=default_value)
            return np.maximum(result, epsilon)
        elif isinstance(arr, torch.Tensor):
            if arr.numel() == 0:
                return torch.tensor(default_value, dtype=arr.dtype, device=arr.device)
            result = torch.std(arr, dim=axis) if axis is not None else torch.std(arr)
            result = sanitize_array(
                result,
                nan_replace=default_value,
                posinf_replace=default_value,
                neginf_replace=default_value
            )
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=arr.dtype, device=arr.device)
            # Ensure minimum value - keep dtype/device aligned with result
            min_val = torch.tensor(epsilon, dtype=result.dtype, device=result.device)
            return torch.maximum(result, min_val)

        else:
            return default_value

    except Exception:
        return default_value


def safe_linear_slope(
    x: Union[np.ndarray, list, tuple],
    y: Union[np.ndarray, list, tuple],
    default_value: float = 0.0,
    min_points: int = 2,
    epsilon: float = 1e-8
) -> float:
    """
    Compute slope of y vs x robustly without np.polyfit/LAPACK calls.

    This avoids downstream linear algebra failures when inputs contain NaN/Inf,
    have near-zero variance, or are otherwise degenerate.
    """
    try:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

        if x_arr.size != y_arr.size:
            n = min(x_arr.size, y_arr.size)
            x_arr = x_arr[:n]
            y_arr = y_arr[:n]

        finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]

        if x_arr.size < min_points:
            return float(default_value)

        x_mean = float(np.mean(x_arr))
        y_mean = float(np.mean(y_arr))
        x_centered = x_arr - x_mean
        y_centered = y_arr - y_mean

        denom = float(np.dot(x_centered, x_centered))
        if not np.isfinite(denom) or denom <= epsilon:
            return float(default_value)

        numer = float(np.dot(x_centered, y_centered))
        slope = numer / denom
        if not np.isfinite(slope):
            return float(default_value)

        return float(slope)
    except Exception:
        return float(default_value)


def safe_zscore(
    arr: Union[np.ndarray, torch.Tensor],
    axis: Optional[int] = None,
    epsilon: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Safe z-score normalization: (x - mean) / (std + epsilon)
    
    Args:
        arr: Input array
        axis: Axis along which to compute
        epsilon: Added to std to prevent division by zero
    
    Returns:
        Z-scored array
    """
    mean_val = safe_mean(arr, axis=axis, default_value=0.0)
    std_val = safe_std(arr, axis=axis, epsilon=epsilon, default_value=1.0)
    
    if isinstance(arr, np.ndarray):
        return (arr - mean_val) / std_val
    elif isinstance(arr, torch.Tensor):
        return (arr - mean_val) / std_val
    else:
        return arr


# Convenience function for genome operations
def sanitize_genome_weights(genome: Any) -> None:
    """
    Sanitize all weights in a genome to ensure numerical stability.
    
    This should be called after mutation, crossover, or any weight modification.
    
    Args:
        genome: EvolvableGenome instance
    """
    if not hasattr(genome, 'genes'):
        return
    
    for gene in genome.genes:
        # Sanitize weights
        if hasattr(gene, 'weights') and gene.weights is not None:
            gene.weights = sanitize_array(gene.weights)
        
        # Sanitize bias
        if hasattr(gene, 'bias') and gene.bias is not None:
            gene.bias = sanitize_array(gene.bias)
        
        # Sanitize plasticity
        if hasattr(gene, 'plasticity') and gene.plasticity is not None:
            gene.plasticity = sanitize_array(gene.plasticity)
        
        # Sanitize batch norm parameters
        if hasattr(gene, 'bn_gamma') and gene.bn_gamma is not None:
            gene.bn_gamma = sanitize_array(gene.bn_gamma)
        if hasattr(gene, 'bn_beta') and gene.bn_beta is not None:
            gene.bn_beta = sanitize_array(gene.bn_beta)
        if hasattr(gene, 'bn_running_mean') and gene.bn_running_mean is not None:
            gene.bn_running_mean = sanitize_array(gene.bn_running_mean)
        if hasattr(gene, 'bn_running_var') and gene.bn_running_var is not None:
            gene.bn_running_var = sanitize_array(gene.bn_running_var, nan_replace=1.0, posinf_replace=1.0, neginf_replace=1.0)
        
        # Sanitize layer norm parameters
        if hasattr(gene, 'ln_gamma') and gene.ln_gamma is not None:
            gene.ln_gamma = sanitize_array(gene.ln_gamma)
        if hasattr(gene, 'ln_beta') and gene.ln_beta is not None:
            gene.ln_beta = sanitize_array(gene.ln_beta)


# Global flag for debugging
DEBUG_NUMERIC_SAFETY = False

def set_debug_mode(enabled: bool = True) -> None:
    """Enable/disable debug mode for numeric safety checks."""
    global DEBUG_NUMERIC_SAFETY
    DEBUG_NUMERIC_SAFETY = enabled
