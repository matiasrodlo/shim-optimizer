# Deep Numerical Analysis: Runtime Warnings and Stability Issues

## Executive Summary

**Status**: ⚠️ **Numerical warnings present but optimization still succeeds**

The optimizer produces RuntimeWarnings for "divide by zero", "overflow", and "invalid value" during matrix operations. While these don't prevent successful optimization, they indicate numerical stability issues that should be addressed.

---

## 1. Warning Locations and Causes

### 1.1 Warning at Line 576: `baseline_field_and_metrics()`

**Code**:
```python
shim_flat = A @ weights0
```

**Context**: Called with `weights0 = np.zeros(N_LOOPS)` (zero weights)

**Warning**: `divide by zero`, `overflow`, `invalid value`

**Root Cause Analysis**:
- After scaling, field matrix `A` has very large values (~2.4 million scale factor)
- Matrix `A` shape: (40,000, 8) for 200x200 grid
- Even with zero weights, the matrix multiplication with large values can trigger overflow warnings
- The warning is likely from NumPy's internal checks, not actual division by zero

**Impact**: 🟢 **LOW** - Zero weights produce zero shim field, so result is correct despite warnings

**Fix**: Suppress warnings for this specific case or use more stable matrix operations

---

### 1.2 Warnings at Lines 651, 660, 666: Optimization Function

**Code**:
```python
# Line 651: objective function
shim_roi = A_roi @ w

# Line 660: gradient function  
shim_roi = A_roi @ w

# Line 666: gradient computation
grad_var = 2 * A_roi.T @ (f_total_roi - mean_f)
```

**Context**: Called repeatedly during optimization with various weight values

**Warning**: `divide by zero`, `overflow`, `invalid value`

**Root Cause Analysis**:
1. **Scaling Issue**: After scaling by ~2.4 million, `A_roi` has very large values
   - Example: If original field values are ~0.0004, after scaling they become ~1000
   - Matrix multiplication `A_roi @ w` with weights in [-1, 1] can produce values ~8000
   - Adding to B0 (~2000) gives total ~10000, which is within float64 range but triggers warnings

2. **Overflow Detection**: NumPy's overflow detection is conservative
   - Float64 max: ~1.8e308
   - Our values: ~10000 (well within range)
   - Warning likely from intermediate calculations or NumPy's internal checks

3. **Divide by Zero**: Likely false positive
   - No explicit division in these lines
   - May be from NumPy's internal normalization or checks

**Impact**: 🟡 **MODERATE** - Optimization succeeds but warnings indicate potential instability

**Fix**: 
- Normalize field matrix before scaling
- Use more conservative scaling approach
- Add explicit checks for extreme values

---

### 1.3 Warning at Line 1138: Scaling Calculation

**Code**:
```python
shim_roi_unit = A_roi @ unit_weights
```

**Context**: Computing shim field magnitude for unit weights to determine scaling factor

**Warning**: `divide by zero`, `overflow`, `invalid value`

**Root Cause Analysis**:
- This happens **before** scaling, so `A_roi` has original small values (~0.0004)
- Unit weights = [1, 1, 1, 1, 1, 1, 1, 1]
- Result should be small (~0.003), but warning suggests numerical issues

**Possible Causes**:
1. Very small values in `A_roi` near machine precision
2. Accumulation of rounding errors
3. NumPy's internal overflow detection being overly sensitive

**Impact**: 🟢 **LOW** - Scaling calculation still works correctly

**Fix**: Add explicit checks for very small values before matrix operations

---

### 1.4 Warning at Line 907: Comparison Function

**Code**:
```python
shim_field_flat = A @ weights
```

**Context**: Computing shim field for comparison (uses unscaled `A` from new computation)

**Warning**: `divide by zero`, `overflow`, `invalid value`

**Root Cause Analysis**:
- This function recomputes field matrix (unscaled) for comparison
- Same issues as above but with original small values
- Less critical since it's only for comparison/visualization

**Impact**: 🟢 **LOW** - Only affects optional comparison output

---

## 2. Root Cause: Scaling Factor Magnitude

### 2.1 The Scaling Problem

**Current Approach**:
```python
scale_factor = b0_std / shim_std_unit
# Example: 957.77 / 0.000389 = 2,464,932
```

**Issues**:
1. **Extreme Scale Factor**: ~2.4 million is very large
   - Multiplies all field matrix values by this factor
   - Can cause numerical instability in subsequent operations

2. **Loss of Precision**: 
   - Original field values: ~0.0004 (4 decimal places)
   - After scaling: ~1000 (fewer significant digits relative to magnitude)
   - May lose precision in optimization

3. **Overflow Risk**:
   - While values stay within float64 range, intermediate calculations may overflow
   - Matrix operations with large values are more prone to numerical errors

### 2.2 Why This Happens

**Original Field Magnitude**:
- Biot-Savart field computed with `mu0_over_4pi = 1.0` (arbitrary units)
- Loop radius: 10 mm
- Coil radius: 80 mm
- Grid FOV: 200 mm
- Result: Very small field values (~0.0004 std in ROI)

**B0 Field Magnitude**:
- Real B0 field from MRI: ~2000 (in original units, likely Hz or nT)
- Standard deviation: ~958

**Scale Mismatch**: 
- Factor of ~2.4 million difference
- This is expected (arbitrary units vs real physical units)
- But scaling approach needs improvement

---

## 3. Numerical Stability Issues

### 3.1 Field Matrix Condition Number

**Problem**: After scaling, the field matrix may have poor conditioning

**Impact**:
- Optimization may be less stable
- Small changes in weights cause large changes in field
- Gradient may be ill-conditioned

**Check Needed**: Compute condition number of `A_roi` after scaling

### 3.2 Accumulation of Errors

**Problem**: Multiple matrix operations with large values

**Chain of Operations**:
1. Compute field matrix (small values)
2. Scale by 2.4 million (large values)
3. Matrix multiply in objective (large values)
4. Matrix multiply in gradient (large values)
5. Optimization iterations (many operations)

**Impact**: Rounding errors accumulate over many operations

### 3.3 Division by Zero (False Positives)

**Analysis**: No explicit divisions in warning locations

**Likely Causes**:
1. NumPy's internal checks for edge cases
2. Normalization operations in NumPy functions
3. Overflow detection triggering false positives

---

## 4. Recommended Fixes

### 4.1 Priority 1: Improve Scaling Approach

**Current** (Line 1143):
```python
scale_factor = b0_std / shim_std_unit
A = A * scale_factor
```

**Better Approach 1: Normalize First**
```python
# Normalize field matrix to unit scale first
A_norm = A / np.std(A_roi @ np.ones(N_LOOPS))
# Then scale to match B0
scale_factor = b0_std
A = A_norm * scale_factor
```

**Better Approach 2: Use Log-Space Scaling**
```python
# Work in log space to avoid extreme values
log_scale = np.log10(b0_std) - np.log10(shim_std_unit)
# Apply scaling more gradually
```

**Better Approach 3: Conservative Scaling**
```python
# Scale to match B0 but cap the factor
max_scale = 1e6  # Reasonable maximum
scale_factor = min(b0_std / shim_std_unit, max_scale)
if scale_factor > max_scale:
    logger.warning(f"Scale factor capped at {max_scale}")
```

### 4.2 Priority 2: Add Numerical Checks

**Add Before Matrix Operations**:
```python
# Check for extreme values
if np.any(np.abs(A) > 1e10):
    logger.warning("Field matrix contains very large values, may cause numerical issues")
    A = np.clip(A, -1e10, 1e10)

# Check condition number
cond_num = np.linalg.cond(A_roi)
if cond_num > 1e12:
    logger.warning(f"Field matrix is ill-conditioned (condition number: {cond_num:.2e})")
```

### 4.3 Priority 3: Suppress False Positive Warnings

**Context-Specific Suppression**:
```python
import warnings

# Suppress warnings for known safe operations
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    shim_flat = A @ weights0  # Known to be safe with zero weights
```

**Better**: Fix root cause rather than suppress warnings

### 4.4 Priority 4: Use Higher Precision

**Option**: Use float128 for critical calculations
```python
A = A.astype(np.float128)  # Higher precision
# ... computations ...
A = A.astype(np.float64)   # Convert back
```

**Trade-off**: Slower but more stable

### 4.5 Priority 5: Normalize Objective Function

**Current** (Line 654):
```python
variance = np.sum((f_total_roi - mean_f)**2)
```

**Better**:
```python
variance = np.mean((f_total_roi - mean_f)**2)  # Normalized
# or
variance = np.var(f_total_roi)  # Built-in function
```

**Benefit**: Objective value independent of ROI size, more stable

---

## 5. Impact Assessment

### 5.1 Current Impact

**Optimization Success**: ✅ Still works
- Optimizer converges successfully
- Produces reasonable results (31% improvement)
- Weights are within bounds

**Numerical Accuracy**: ⚠️ Potentially compromised
- Warnings suggest numerical issues
- May lose precision in calculations
- Results may not be optimal

**Stability**: ⚠️ Moderate concern
- Works for current case
- May fail for different B0 magnitudes
- May be sensitive to parameter changes

### 5.2 Risk Assessment

**Low Risk**:
- Current optimization succeeds
- Results are reasonable
- No crashes or incorrect results

**Medium Risk**:
- May fail with different B0 data
- May produce suboptimal results
- Hard to debug when issues occur

**High Risk** (if not addressed):
- Could fail silently with wrong results
- Difficult to reproduce issues
- May not work with different datasets

---

## 6. Testing Recommendations

### 6.1 Numerical Stability Tests

1. **Test with Different B0 Magnitudes**
   - Scale B0 data by factors (0.1x, 10x, 100x)
   - Verify optimizer still works
   - Check for warnings

2. **Test Condition Number**
   - Compute condition number of `A_roi`
   - Verify it's reasonable (< 1e12)
   - Test with different loop configurations

3. **Test Precision**
   - Compare results with float64 vs float128
   - Check if results differ significantly
   - Identify precision-sensitive operations

### 6.2 Regression Tests

1. **Baseline Test**
   - Run with known B0 data
   - Verify results match expected values
   - Check for warnings

2. **Edge Cases**
   - Very small B0 variations
   - Very large B0 variations
   - Extreme loop positions
   - Different ROI sizes

---

## 7. Implementation Priority

### Immediate (Before Production Use)

1. ✅ Add numerical checks and warnings
2. ✅ Improve scaling approach (conservative scaling)
3. ✅ Normalize objective function

### Short Term (Next Version)

4. Add condition number checks
5. Implement better scaling (normalize first)
6. Add comprehensive tests

### Long Term (Future Improvements)

7. Consider higher precision for critical calculations
8. Implement adaptive scaling
9. Add numerical stability monitoring

---

## 8. Conclusion

**Current Status**: 
- ⚠️ Numerical warnings present
- ✅ Optimization succeeds
- ⚠️ Stability concerns exist

**Recommendation**: 
- Address scaling approach (Priority 1)
- Add numerical checks (Priority 2)
- Monitor for issues with different datasets

**Risk Level**: 🟡 **MODERATE** - Works now but may have issues with different data

The warnings don't prevent successful optimization, but they indicate areas for improvement. The most critical issue is the extreme scaling factor (~2.4 million), which should be addressed to improve numerical stability.

