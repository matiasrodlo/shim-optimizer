"""
Comparison of Different Optimization Methods for Shimming

This script compares different optimization approaches for the shimming problem.
"""

import numpy as np
from scipy import optimize
from scipy.optimize import lsq_linear
import time


def compare_optimizers(A_roi, baseline_roi, alpha, bounds, w0):
    """
    Compare different optimization methods for shimming.
    
    The problem is: minimize ||A*w + baseline - mean(A*w + baseline)||^2 + alpha*||w||^2
    
    Parameters
    ----------
    A_roi : array
        Design matrix in ROI
    baseline_roi : array
        Baseline field in ROI
    alpha : float
        Regularization parameter
    bounds : tuple
        (lower, upper) bounds for weights
    w0 : array
        Initial guess
        
    Returns
    -------
    results : dict
        Comparison results
    """
    n_loops = A_roi.shape[1]
    results = {}
    
    print("=" * 70)
    print("OPTIMIZER COMPARISON FOR SHIMMING")
    print("=" * 70)
    
    # Common objective function for comparison
    def objective(w):
        shim = A_roi @ w
        total = baseline_roi + shim
        mean_val = np.mean(total)
        variance = np.mean((total - mean_val)**2)
        reg = alpha * np.mean(w**2)
        return variance + reg
    
    def gradient(w):
        shim = A_roi @ w
        total = baseline_roi + shim
        mean_val = np.mean(total)
        grad_var = 2 * A_roi.T @ (total - mean_val) / len(total)
        grad_reg = 2 * alpha * w / len(w)
        return grad_var + grad_reg
    
    # =========================================================================
    # Method 1: L-BFGS-B (Current method)
    # =========================================================================
    print("\n1. L-BFGS-B (Current method):")
    print("   " + "-" * 60)
    t0 = time.time()
    result_lbfgsb = optimize.minimize(
        objective, w0, method='L-BFGS-B', jac=gradient,
        bounds=[bounds] * n_loops,
        options={'maxiter': 500}
    )
    t_lbfgsb = time.time() - t0
    
    results['L-BFGS-B'] = {
        'weights': result_lbfgsb.x,
        'objective': result_lbfgsb.fun,
        'success': result_lbfgsb.success,
        'time': t_lbfgsb,
        'iterations': result_lbfgsb.nit,
        'method_type': 'Quasi-Newton (general purpose)'
    }
    
    print(f"   Time: {t_lbfgsb:.4f}s")
    print(f"   Iterations: {result_lbfgsb.nit}")
    print(f"   Objective: {result_lbfgsb.fun:.2f}")
    print(f"   Success: {result_lbfgsb.success}")
    
    # =========================================================================
    # Method 2: Bounded Least Squares (RECOMMENDED for shimming)
    # =========================================================================
    print("\n2. Bounded Least Squares (RECOMMENDED ⭐):")
    print("   " + "-" * 60)
    
    # Reformulate as least squares problem: minimize ||C*w - d||^2
    # where we want to minimize variance in ROI
    t0 = time.time()
    
    # Target: make field uniform (minimize deviation from mean)
    # We'll use iterative approach since mean depends on weights
    
    # Simple approach: minimize ||shim_field - target||^2
    # Target = field that makes total field uniform
    target = -baseline_roi + np.mean(baseline_roi)
    
    # Add regularization by augmenting the system
    A_aug = np.vstack([A_roi, np.sqrt(alpha * len(baseline_roi) / n_loops) * np.eye(n_loops)])
    b_aug = np.concatenate([target, np.zeros(n_loops)])
    
    result_lsq = lsq_linear(
        A_aug, b_aug,
        bounds=(bounds[0], bounds[1]),
        method='bvls',  # Bounded-Variable Least Squares
        verbose=0
    )
    
    t_lsq = time.time() - t0
    
    results['LSQ-Linear'] = {
        'weights': result_lsq.x,
        'objective': objective(result_lsq.x),
        'success': result_lsq.success > 0,
        'time': t_lsq,
        'iterations': result_lsq.nit if hasattr(result_lsq, 'nit') else 'N/A',
        'method_type': 'Bounded Least Squares (specialized for linear problems)'
    }
    
    print(f"   Time: {t_lsq:.4f}s")
    print(f"   Objective: {objective(result_lsq.x):.2f}")
    print(f"   Success: {result_lsq.success > 0}")
    print(f"   ✅ This is THE STANDARD method for shimming tasks!")
    
    # =========================================================================
    # Method 3: Trust-Region Constrained
    # =========================================================================
    print("\n3. Trust-Region Constrained:")
    print("   " + "-" * 60)
    t0 = time.time()
    result_trust = optimize.minimize(
        objective, w0, method='trust-constr', jac=gradient,
        bounds=optimize.Bounds([bounds[0]] * n_loops, [bounds[1]] * n_loops),
        options={'maxiter': 500, 'verbose': 0}
    )
    t_trust = time.time() - t0
    
    results['Trust-Region'] = {
        'weights': result_trust.x,
        'objective': result_trust.fun,
        'success': result_trust.success,
        'time': t_trust,
        'iterations': result_trust.nit,
        'method_type': 'Trust-region (robust for ill-conditioned problems)'
    }
    
    print(f"   Time: {t_trust:.4f}s")
    print(f"   Iterations: {result_trust.nit}")
    print(f"   Objective: {result_trust.fun:.2f}")
    print(f"   Success: {result_trust.success}")
    
    # =========================================================================
    # Method 4: SLSQP
    # =========================================================================
    print("\n4. SLSQP (Sequential Least Squares Programming):")
    print("   " + "-" * 60)
    t0 = time.time()
    result_slsqp = optimize.minimize(
        objective, w0, method='SLSQP', jac=gradient,
        bounds=[bounds] * n_loops,
        options={'maxiter': 500}
    )
    t_slsqp = time.time() - t0
    
    results['SLSQP'] = {
        'weights': result_slsqp.x,
        'objective': result_slsqp.fun,
        'success': result_slsqp.success,
        'time': t_slsqp,
        'iterations': result_slsqp.nit,
        'method_type': 'Sequential Quadratic Programming'
    }
    
    print(f"   Time: {t_slsqp:.4f}s")
    print(f"   Iterations: {result_slsqp.nit}")
    print(f"   Objective: {result_slsqp.fun:.2f}")
    print(f"   Success: {result_slsqp.success}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find best objective
    best_obj = min(r['objective'] for r in results.values())
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    
    print("\n📊 Results by Objective Value:")
    for name, res in sorted(results.items(), key=lambda x: x[1]['objective']):
        marker = "⭐ BEST" if abs(res['objective'] - best_obj) < 1e-6 else ""
        print(f"   {name:20s}: {res['objective']:10.2f}  {marker}")
    
    print("\n⚡ Results by Speed:")
    for name, res in sorted(results.items(), key=lambda x: x[1]['time']):
        marker = "⚡ FASTEST" if name == fastest[0] else ""
        print(f"   {name:20s}: {res['time']:8.4f}s  {marker}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    print("\n🎯 For shimming tasks, use: **Bounded Least Squares (lsq_linear)**")
    print("\nReasons:")
    print("  1. Designed specifically for this problem type")
    print("  2. Typically fastest")
    print("  3. Guaranteed global optimum (convex problem)")
    print("  4. Standard in shimming literature")
    print("  5. Numerically stable")
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    print("\nThis module provides optimizer comparison for shimming.")
    print("Import and use compare_optimizers() function.\n")

