#!/bin/bash
#
# Run LSQ Shim Optimizer (Standard Method)
#
# This uses scipy.optimize.lsq_linear with Bounded-Variable Least Squares (BVLS)
# which is the STANDARD method for shimming tasks in the literature.
#

PYTHON=/usr/bin/python3

# Check if dataset exists
if [ ! -d "../dataset" ]; then
    echo "Error: Dataset directory not found at ../dataset"
    echo "Please ensure the dataset is in the correct location"
    exit 1
fi

echo "=============================================="
echo "LSQ Shim Optimizer (Standard Method)"
echo "Method: Bounded Least Squares (BVLS)"
echo "=============================================="
echo ""

# Run with provided arguments or defaults
$PYTHON shim_optimizer_lsq.py "$@"

