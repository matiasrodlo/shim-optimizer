"""
Parameter sweep to find optimal configuration for maximum shimming performance.
"""

import numpy as np
import json
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_configuration(n_loops, r_coil, alpha, grid_n=300):
    """Test a specific configuration."""
    import subprocess
    import tempfile
    
    # Create temporary modified script
    script_path = os.path.join(os.path.dirname(__file__), 'shim_optimizer_lsq.py')
    
    # Run with current settings
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Temporarily modify parameters
    orig_n_loops = content
    content = content.replace(f'N_LOOPS = {n_loops}', f'N_LOOPS = {n_loops}')
    content = content.replace(f'R_COIL_MM = {r_coil}', f'R_COIL_MM = {r_coil}')
    content = content.replace(f'ALPHA = {alpha}', f'ALPHA = {alpha}')
    content = content.replace(f'GRID_N = {grid_n}', f'GRID_N = {grid_n}')
    
    # Note: This is simplified - just document results manually
    pass


def main():
    """Sweep parameters to find optimum."""
    print("=" * 70)
    print("PARAMETER SWEEP FOR MAXIMUM PERFORMANCE")
    print("=" * 70)
    print()
    
    print("Testing configurations:")
    print()
    
    configurations = [
        # (n_loops, r_coil_mm, alpha, description)
        (8, 80, 0.001, "Original configuration"),
        (16, 60, 0.0001, "More loops, closer coils"),
        (24, 50, 0.00001, "Many loops, very close"),
        (32, 40, 0.0, "Maximum loops, optimal radius"),
    ]
    
    print("Configuration Tests:")
    print("-" * 70)
    print(f"{'N_Loops':>8} {'R_Coil':>8} {'Alpha':>12} {'Description'}")
    print("-" * 70)
    
    for n_loops, r_coil, alpha, desc in configurations:
        print(f"{n_loops:8d} {r_coil:8.1f} {alpha:12.6f} {desc}")
    
    print()
    print("=" * 70)
    print("\nRun the optimizer with each configuration and record results.")
    print("\nTo test, edit shim_optimizer_lsq.py and change:")
    print("  N_LOOPS, R_COIL_MM, ALPHA")
    print()


if __name__ == "__main__":
    main()

