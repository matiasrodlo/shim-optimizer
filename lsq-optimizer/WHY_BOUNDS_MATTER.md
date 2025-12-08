# Why Bounds Matter in Optimization

## Your Question: "Why doesn't the optimizer find the best point automatically?"

**Short Answer**: The optimizer IS working perfectly - but you gave it constraints (bounds) that prevent it from finding the true optimum.

## The Key Concept: Constrained vs Unconstrained Optimization

### What You're Asking the Optimizer:

```python
BOUNDS = (-1.0, 1.0)
```

Translates to:

> "Find the best shimming solution, but ALL weights must be between -1 and +1"

### What the Optimizer Does:

The optimizer **perfectly solves the problem you gave it**:
- It finds the BEST solution within (-1, 1) bounds
- It pushes weights to ±1 (the limits you set)
- It achieves 3.13% improvement (the best possible with these constraints)

### The Problem:

The **TRUE optimal solution** needs weights like:
- +45.15, -48.14, +62.54, -25.37, etc.

But you told it "stay between -1 and +1", so it CAN'T use these values!

---

## Why This Happened

### In the Original Code (savart-optimizer):

```python
BOUNDS = (-1.0, 1.0)  # Too small!
```

**Why was it set this small?**

1. **Conservative default**: Safe starting point
2. **Not based on physical limits**: Just arbitrary
3. **Assumption**: "weights should be small normalized values"
4. **Problem**: Not realistic for this shimming task

### The True Optimum:

```python
BOUNDS = (-100, 100)
Optimal weights: [+45, -48, +62, -25, +27, -39, -16, -6]
Improvement: 25.92%
```

---

## How to Choose Bounds Correctly

### Method 1: Start Large, Then Constrain (RECOMMENDED)

```python
# Step 1: Find unconstrained optimum
BOUNDS = (-100, 100)  # or even (-1000, 1000)
# Run optimizer
# Result: max weight is +62.54

# Step 2: Check if this is physically feasible
# Question: Can our coils handle 62 Amps?
# - If YES: Keep large bounds
# - If NO: Use smaller bounds (but accept lower performance)
```

### Method 2: Use Physical Constraints

```python
# If you know your coils can handle ±50 Amps:
BOUNDS = (-50.0, 50.0)

# If power supply limited to ±20 Amps:
BOUNDS = (-20.0, 20.0)
```

### Method 3: Check the Warnings

The optimizer TELLS you when bounds are too tight:

```
⚠️  "8 weights at bounds (may indicate suboptimal solution)"
```

This means: **"I want to go further but you won't let me!"**

---

## Real-World Example: Your Results

| Bounds | Improvement | Weights at Bounds | Interpretation |
|--------|-------------|-------------------|----------------|
| (-1, 1) | 3.13% | 8/8 (100%) | ❌ Way too tight |
| (-5, 5) | 13.72% | 8/8 (100%) | ❌ Still too tight |
| (-20, 20) | 24.95% | 2/8 (25%) | ⚠️ Almost there |
| (-100, 100) | 25.92% | 0/8 (0%) | ✅ Optimal! |

### What This Tells Us:

1. **True optimum needs weights up to ±62**
2. **Bounds < 62 artificially limit performance**
3. **Bounds ≥ 62 give full performance**

---

## Why Not Just Remove Bounds Entirely?

Good question! Here's why we keep them:

### Without Bounds:

```python
# Mathematically unconstrained
# Problem: May give unrealistic solutions
# - Weights could be ±10,000
# - Not physically realizable
# - Numerical instability
```

### With Reasonable Bounds:

```python
BOUNDS = (-100, 100)
# Benefits:
# - Prevents extreme numerical values
# - Ensures physical feasibility
# - Catches errors (if optimizer wants weight=10,000, something's wrong!)
# - Documentation of hardware limits
```

---

## The Role of Regularization vs Bounds

### Regularization (α):

```python
ALPHA = 0.001
# Effect: Penalizes large weights in the objective function
# Nature: "Soft" constraint (can be violated if beneficial)
# Purpose: Prefer smaller weights, but allow large if needed
```

### Bounds:

```python
BOUNDS = (-100, 100)
# Effect: HARD limit - cannot exceed
# Nature: Absolute constraint
# Purpose: Physical/safety limits
```

### Why You Need BOTH:

- **α**: Encourages small weights (energy efficiency)
- **Bounds**: Enforces physical limits

---

## How the Optimizer Actually Works

### Constrained Optimization Problem:

```
minimize: f(w) = variance + α*||w||²
subject to: lower ≤ w ≤ upper
```

### What Happens:

1. **Interior Solution**: If optimum is within bounds
   ```
   Optimal w = [+45, -48, +62, ...]
   Bounds = (-100, 100)
   Result: All weights in interior ✅
   ```

2. **Boundary Solution**: If optimum is outside bounds
   ```
   Optimal w = [+62, ...]  # Would like to go to +80
   Bounds = (-50, 50)
   Result: Weight clamped at +50 (at bound) ⚠️
   ```

### The Warning System:

```python
if weight ≈ bound:
    logger.warning("Weight at bound - solution may be suboptimal")
```

This is the optimizer saying: **"I'd go further if you let me!"**

---

## Practical Recommendation

### For Your Shimming Task:

1. **Simulation/Testing Phase**:
   ```python
   BOUNDS = (-100, 100)  # Large bounds to find true optimum
   ```

2. **Check Results**:
   ```python
   # Max weight found: +62.54
   # → You need coils that can handle ±65 Amps (with margin)
   ```

3. **Production Phase**:
   ```python
   # If your hardware can handle ±50 Amps:
   BOUNDS = (-50, 50)
   # Expected: ~25% improvement (slightly less than 25.92%)
   
   # If your hardware can only handle ±20 Amps:
   BOUNDS = (-20, 20)
   # Expected: ~24.95% improvement (more limited)
   ```

---

## Summary

### Question: "Why doesn't optimizer find best point automatically?"

**Answer**: 

**It DOES find the best point - but only within the constraints YOU gave it!**

- **BOUNDS = (-1, 1)**: Best point with tiny weights → 3.13%
- **BOUNDS = (-100, 100)**: Best point without constraint → 25.92%

### The Real Question Should Be:

**"What bounds should I use?"**

**Answer**:
1. **For finding optimal performance**: Use large bounds (-100, 100)
2. **For real hardware**: Use physical limits (e.g., -50, 50 if coils can handle 50A)
3. **Check warnings**: If weights hit bounds → bounds too tight!

### Key Insight:

**The optimizer is like a GPS**:
- You say: "Get me to downtown, but don't use highways"
- GPS finds best route without highways
- But it's not the FASTEST route (highway route would be faster)
- If you remove the "no highway" constraint → faster route found

**Same with bounds**:
- You say: "Optimize shimming, but weights must be ±1"  
- Optimizer finds best solution with ±1 weights
- But it's not the BEST solution (larger weights would be better)
- If you remove the tight constraint → better solution found

---

## The Code Change You Need:

### Before (Arbitrary):
```python
BOUNDS = (-1.0, 1.0)  # Why? No physical reason!
```

### After (Physical):
```python
# Option 1: Find true optimum first
BOUNDS = (-100.0, 100.0)  
# Run, see max is +62, then decide if hardware can handle it

# Option 2: Use known hardware limits
BOUNDS = (-50.0, 50.0)  # Our coils can handle 50 Amps
```

---

## Final Thought:

**The optimizer isn't broken - the constraints were!**

Always ask: **"Are my bounds based on physics, or just arbitrary?"**

If arbitrary → use larger bounds to find true optimum!

