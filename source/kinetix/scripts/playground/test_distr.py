"""Script to test if initialized values follow a uniform distribution."""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
from scipy import stats

# Set random seed for reproducibility
key = jr.PRNGKey(42)

# Initialization parameters
init_min = -1.0
init_max = 1.0
elite_popsize = 10  # Number of individuals
num_dims = 1000  # Number of parameters per individual

print(f"Testing uniform distribution initialization:")
print(f"  Range: [{init_min}, {init_max}]")
print(f"  Population size: {elite_popsize}")
print(f"  Dimensions per individual: {num_dims}")
print()

# Generate uniform random values
key, subkey = jr.split(key)
initialization = jax.random.uniform(
    subkey,
    (elite_popsize, num_dims),
    minval=init_min,
    maxval=init_max,
)

# Convert to numpy for statistical testing
initialization_np = onp.array(initialization)

print("Statistical Tests:")
print("=" * 60)

# Test each individual
pvalues = []
for idx in range(elite_popsize):
    individual_weights = initialization_np[idx, :]
    
    # Get observed min/max
    min_w = individual_weights.min()
    max_w = individual_weights.max()
    
    # Normalize to [0, 1] for uniform test
    normalized = (individual_weights - min_w) / (max_w - min_w + 1e-10)
    
    # Kolmogorov-Smirnov test against uniform distribution
    stat, pvalue = stats.kstest(normalized, 'uniform')
    pvalues.append(pvalue)
    
    print(f"Individual {idx}:")
    print(f"  Observed range: [{min_w:.4f}, {max_w:.4f}]")
    print(f"  KS statistic: {stat:.4f}")
    print(f"  p-value: {pvalue:.4f}")
    print(f"  Result: {'UNIFORM ✓' if pvalue > 0.05 else 'NOT UNIFORM ✗'}")
    print()

# Summary statistics
print("=" * 60)
print("Summary:")
print(f"  Mean p-value: {onp.mean(pvalues):.4f}")
print(f"  Min p-value: {onp.min(pvalues):.4f}")
print(f"  Max p-value: {onp.max(pvalues):.4f}")
print(f"  Individuals classified as uniform: {sum(1 for p in pvalues if p > 0.05)}/{elite_popsize}")

# Additional checks
print()
print("Additional Statistics:")
print("=" * 60)

# Check if values are within bounds
all_values = initialization_np.flatten()
within_bounds = ((all_values >= init_min) & (all_values <= init_max)).all()
print(f"  All values within [{init_min}, {init_max}]: {within_bounds}")

# Expected variance for uniform distribution
expected_var = (init_max - init_min) ** 2 / 12.0
actual_var = onp.var(all_values)
print(f"  Expected variance (theoretical): {expected_var:.6f}")
print(f"  Actual variance (observed): {actual_var:.6f}")
print(f"  Variance ratio: {actual_var / expected_var:.4f}")

# Check mean (should be close to (min+max)/2)
expected_mean = (init_min + init_max) / 2.0
actual_mean = onp.mean(all_values)
print(f"  Expected mean: {expected_mean:.4f}")
print(f"  Actual mean: {actual_mean:.4f}")
print(f"  Mean difference: {abs(actual_mean - expected_mean):.6f}")
