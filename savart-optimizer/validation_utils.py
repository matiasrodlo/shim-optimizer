"""
Validation utilities for shim optimization results.

Provides functions to validate optimization results and assess solution quality.
"""

import numpy as np
import logging


def validate_optimization_result(w_opt, A, baseline_b0, roi_mask, bounds, history, logger=None):
    """
    Validate optimization result quality and identify potential issues.
    
    Parameters
    ----------
    w_opt : ndarray
        Optimized weights
    A : ndarray
        Field matrix (flattened)
    baseline_b0 : ndarray
        Baseline B0 field
    roi_mask : ndarray
        ROI mask
    bounds : tuple
        Weight bounds (lower, upper)
    history : dict
        Optimization history
    logger : logging.Logger, optional
        Logger instance
    
    Returns
    -------
    validation_report : dict
        Dictionary containing validation results and warnings
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    report = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'metrics': {}
    }
    
    # Check 1: Weights within bounds
    lower, upper = bounds
    n_at_lower = np.sum(w_opt <= lower + 1e-6)
    n_at_upper = np.sum(w_opt >= upper - 1e-6)
    
    report['metrics']['n_at_lower_bound'] = int(n_at_lower)
    report['metrics']['n_at_upper_bound'] = int(n_at_upper)
    
    if n_at_lower > 0 or n_at_upper > 0:
        report['warnings'].append(
            f"{n_at_lower + n_at_upper} weights at bounds (may indicate suboptimal solution)"
        )
        logger.warning(f"  {n_at_lower} weights at lower bound, {n_at_upper} at upper bound")
    
    # Check 2: Field computation successful
    field = baseline_b0 + (A @ w_opt).reshape(baseline_b0.shape)
    field_finite = np.all(np.isfinite(field))
    
    report['metrics']['field_finite'] = field_finite
    
    if not field_finite:
        report['errors'].append("Optimized field contains NaN/Inf values")
        report['passed'] = False
        logger.error("  Optimized field contains non-finite values!")
    
    # Check 3: Improvement is positive
    std_before = np.std(baseline_b0[roi_mask])
    std_after = np.std(field[roi_mask])
    improvement = 100 * (1 - std_after / std_before)
    
    report['metrics']['improvement_percent'] = float(improvement)
    
    if improvement < 0:
        report['warnings'].append(f"Negative improvement ({improvement:.2f}%)")
        logger.warning(f"  Field got worse! Improvement: {improvement:.2f}%")
    elif improvement < 0.1:
        report['warnings'].append(f"Very small improvement ({improvement:.2f}%)")
        logger.warning(f"  Very small improvement: {improvement:.2f}%")
    
    # Check 4: Power requirements reasonable
    total_power = np.sum(np.abs(w_opt))
    max_weight = np.max(np.abs(w_opt))
    rms_weight = np.sqrt(np.mean(w_opt**2))
    
    report['metrics']['total_power'] = float(total_power)
    report['metrics']['max_weight'] = float(max_weight)
    report['metrics']['rms_weight'] = float(rms_weight)
    
    # Check 5: Convergence quality
    if history['success']:
        report['metrics']['converged'] = True
    else:
        report['warnings'].append(f"Optimizer did not converge: {history['message']}")
        logger.warning(f"  Optimizer did not converge: {history['message']}")
        report['metrics']['converged'] = False
    
    # Check 6: Gradient norm (should be small at convergence)
    if history['grad_norm']:
        final_grad_norm = history['grad_norm'][-1]
        report['metrics']['final_gradient_norm'] = float(final_grad_norm)
        
        if final_grad_norm > 1.0:
            report['warnings'].append(f"Large final gradient norm ({final_grad_norm:.2e})")
            logger.warning(f"  Large final gradient norm: {final_grad_norm:.2e}")
    
    # Check 7: Objective reduction
    if history['initial_objective'] > 0:
        obj_reduction = 100 * (1 - history['final_objective'] / history['initial_objective'])
        report['metrics']['objective_reduction_percent'] = float(obj_reduction)
        
        if obj_reduction < 1.0:
            report['warnings'].append(f"Small objective reduction ({obj_reduction:.2f}%)")
    
    # Overall assessment
    if report['errors']:
        report['passed'] = False
        report['quality'] = 'FAILED'
    elif len(report['warnings']) > 3:
        report['quality'] = 'POOR'
    elif len(report['warnings']) > 0:
        report['quality'] = 'ACCEPTABLE'
    else:
        report['quality'] = 'GOOD'
    
    return report


def generate_validation_summary(report, logger=None):
    """
    Generate human-readable validation summary.
    
    Parameters
    ----------
    report : dict
        Validation report from validate_optimization_result
    logger : logging.Logger, optional
        Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Overall Quality: {report['quality']}")
    logger.info(f"Validation Passed: {report['passed']}")
    
    if report['errors']:
        logger.info("\nErrors:")
        for error in report['errors']:
            logger.error(f"  - {error}")
    
    if report['warnings']:
        logger.info("\nWarnings:")
        for warning in report['warnings']:
            logger.warning(f"  - {warning}")
    
    logger.info("\nKey Metrics:")
    for key, value in report['metrics'].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("="*60)

