# CQMOM vs Monte Carlo for 2D Droplet Fragmentation

## Overview

This repository provides a comprehensive comparison between **Conditional Quadrature Method of Moments (CQMOM)** and **Monte Carlo (MC)** methods for simulating droplet fragmentation in 2D. The code implements the Reitz-Diwakar (RD) fragmentation model with advanced statistical methods to track droplet size and velocity distributions during atomization processes.



## Repository Structure

```
├── MTE_2D_Droplet_Breakup.ipynb          # Jupyter notebook version
├── MTE_2D_Droplet_Breakup_python.py      # Python script version  
├── Generate_gif_radius_evolution.py      # GIF animation generator (MC data)
├── Quadrature/
│   └── CQMOM.py                          # CQMOM implementation
├── utils/
│   ├── flux_calculator.py               # Flux calculations for CQMOM
│   └── testing_utils.py                 # Utility functions
└── README.md                            # This file
```

## Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib jupyter tqdm pathlib
```

### Basic Usage

**Option 1: Python Script (Recommended for performance)**
```bash
python MTE_2D_Droplet_Breakup_python.py
```

**Option 2: Jupyter Notebook (Recommended for exploration)**
```bash
jupyter notebook MTE_2D_Droplet_Breakup.ipynb
```

**Option 3: Advanced Visualization (After MC simulation)**
```bash
python Generate_gif_radius_evolution.py
```

> **Performance Note:** The Python script is significantly faster due to true multiprocessing, while the notebook uses threading to avoid compatibility issues.

> **Visualization Note:** The GIF generator requires Monte Carlo data (`radii_timeseries.json`) and creates dynamic animations of size distribution evolution.

## Customization Guide

### 1. Physical Parameters

#### Gas Properties
```python
# Gas phase properties (lines 181-186)
Temperature = 1200.0              # Gas temperature (K)
Pression = 5.0*10**5             # Gas pressure (Pa)
rhog = 1.293*(273.15/Temperature)*(Pression/101325)  # Gas density (kg/m³) - calculated
ug = -20                         # Gas velocity (m/s) - negative = opposing droplet flow
mug = 1.716*(10**(-5))*((Temperature/273.15)**(3/2))*((273.15 + 110.4) / Temperature + 110.4)  # Gas viscosity (Pa.s)
```

#### Liquid Properties
```python
# Liquid phase properties (lines 188-194)
rhol = 800.0                     # Liquid density (kg/m³)
sigmal = 25*10**(-3)             # Surface tension (N/m)
mul = 1.5*10**(-3)               # Liquid dynamic viscosity (Pa.s)
d0 = 2e-3                        # Nozzle diameter (m) - 2 mm
```

**Customizable aspects:**
- **Gas-liquid combination:** Any gas-liquid pair, but ensure physical consistency
- **Temperature range:** Model valid for typical injection conditions (300-1500 K)
- **Pressure conditions:** Suitable for high-pressure injection systems


⚠️ **Important:** The implemented Reitz-Diwakar fragmentation model is specifically designed for **injection/atomization scenarios**. Ensure physical consistency between:
- Weber and Reynolds numbers in realistic ranges (We > 12 for fragmentation)
- Gas-liquid density ratio (was not computed for liquid-liquid)
- Relative velocity conditions (droplet-gas interaction)
- Surface tension and viscosity effects appropriate for the chosen fluids 

### 2. Simulation Settings

```python
# Dimension for the number of moments see next 
dim = 2                          # Dimensions (always 2 for this implementation)
time_step = 1e-6                 # Time step size (s) - decrease for better accuracy
num_particles = 7                # Initial number of droplets
max_nb_fragmentation = 250       # Maximum child droplets per parent droplet
simulation_time = 3000*time_step # Total simulation duration (s)
num_loops = int(simulation_time/time_step)  # Number of time steps

# Performance settings (line 44)
max_workers = 4                  # Number of CPU cores for parallelization
```

**Key considerations:**
- Smaller `time_step` → Better accuracy but longer computation
- Larger `num_particles` → Better statistics but exponential cost increase
- `max_nb_fragmentation` limits memory usage per droplet

### 3. Initial Distribution

```python
# Initial droplet characteristics (lines 200-207)
r0 = d0/2                        # Initial radius (m) - half of nozzle diameter or whatever
u0 = 100                         # Initial velocity (m/s)

# Statistical distribution parameters
mu = np.array([r0, u0])          # Mean values [radius, velocity]
cov = np.diag([(0.10*r0)**2, (0.05*u0)**2])  # Covariance matrix (10% on radius, 5% on speed)

# Moment tracking configuration (lines 209-210)
N = tuple([2 for _ in range(dim)]) # Number of moments per dimension (2x2 = 4 moments total)
```

**Customizable aspects:**
- **Initial size distribution:** Change `(0.10*r0)**2` for different radius variance
- **Initial velocity spread:** Modify `(0.05*u0)**2` for velocity variance  
- **Moment resolution:** Increase `N` values for higher-order moments (computational cost increases)


## Research Extensions

### A. Modifying Fragmentation Models

**Current Implementation:** Reitz-Diwakar model with random process for number of child droplets and their radius (no velocity distribution) for drdt
**Location:** `drdt_with_fragmentation_shrinkage()` (lines 347-428)

**Example: Adding KH_RT model**
```python
def drdt_KH_RT_model(tab_r, tab_u, t, rhog, ug, sigmal, mug):
    pass
```

### B. Custom Child Droplet Distribution

The fragmentation process involves **two separate statistical distributions** that can be customized independently:

#### B.1. Number of Child Droplets Distribution

**Current Implementation:** Log-normal distribution (mode=2, σ=1, range=[1,5])
**Locations:**
- **MC:** Lines 738-754 in `run_simulation_one_droplet()`
- **CQMOM:** Line 559 in `calculate_fragmentation_source_rate()` (fixed value: 3.4 mean value of the distribution of range [2, 6])

```python
# MC: Random number of children per fragmentation event
target_mode = 2
mu_nb = np.log(target_mode)
sigma_nb = 1
nb_min = 1
nb_max = 5

# Sample from log-normal with constraints
while True:
    sample = rng.lognormal(mean=mu_nb, sigma=sigma_nb)
    sample_int = int(np.floor(sample))
    if nb_min <= sample_int <= nb_max:
        nb_gouttes_formees = sample_int
        break
```

**Customization example: Uniform distribution**
```python
# Replace with uniform distribution
nb_gouttes_formees = rng.integers(2, 6)  # Uniform between 2-5 children
```

#### B.2. Child Droplet Size Distribution

**Current Implementation:** Log-normal volume distribution with conservation constraints
**Locations:**
- **MC:** Lines 756-780 in `run_simulation_one_droplet()`
- **CQMOM:** Lines 562-569 in `calculate_fragmentation_source_rate()`

```python
# MC: Volume distribution for each child droplet
volume_k = (4/3)*np.pi*r_node**3
vol_mean = volume_k/(nb_gouttes_formees)
vol_sigma = volume_k/((nb_gouttes_formees)*12)

# Log-normal parameters
mu = np.log(vol_mean**2 / np.sqrt(vol_sigma**2 + vol_mean**2))
sigma_ln = np.sqrt(np.log(1 + (vol_sigma**2 / vol_mean**2)))

# Sample volumes with conservation constraints
vol_candidate = rng.lognormal(mean=mu, sigma=sigma_ln)
```

**Customization example: Beta distribution**
```python
# Replace with Beta distribution for volumes
alpha, beta = 2.0, 3.0
vol_fraction = rng.beta(alpha, beta)
vol_candidate = vol_fraction * volume_k * 0.8  # 80% of parent volume max
```

#### B.3. Consistency Requirements

⚠️ **Critical:** Both MC and CQMOM implementations must use **consistent statistical models**:

**For CQMOM:** Update the integrated term in the source calculation
```python
# In calculate_fragmentation_source_rate()
# Replace lines 568-569 with your custom distribution's moment
res_double_integrale_loi_ln_moments = (u_node**j)*((3/(4*np.pi))**(i/3))*custom_volume_moment
```

**For MC:** Ensure volume conservation and realistic size constraints
```python
# Add validation for your custom distribution
if vol_candidate < min_volume or vol_candidate > max_volume:
    continue  # Reject unrealistic sizes
```

🚨 **Common Pitfall - rchild in drdt function:**

**Location:** Line 381 in `drdt_with_fragmentation_shrinkage()`

When customizing child droplet distributions, you **must** also update the `rchild` parameter used in the radius evolution equation:

```python
# Current implementation (line 381)
rchild = 0.681 * tab_r[i]  # Change the parameters 0.681

# This coefficient comes from: 1 / (mean_nb_children + 1)**(1/3) 
# For current distribution of range [1, 5] : 1/(2.17 + 1)**(1/3) = 0.681
```

**If you change the number of children distribution, update rchild accordingly:**

⚠️ **Ignoring this will lead to inconsistent fragmentation dynamics between the radius evolution (drdt) and the actual fragmentation events!**

### C. New Source Terms for CQMOM

**Location:** `calculate_fragmentation_source_rate()` (lines 539-600)

**Structure:**
```python
def calculate_custom_source_rate(r_node, u_node, i, j):
    """
    Custom source term for moment equations
    
    Parameters:
    - r_node, u_node: Quadrature points
    - i, j: Moment indices
    
    Returns:
    - source_rate: Birth - Death rate
    """
    # Fragmentation criteria
    # Birth rate calculation  
    # Death rate calculation
    # Return net source term
```

### D. Alternative Physical Models

#### Drag Force Models
**Location:** (lines 292-302)
```python
def Cd_custom(rhog, u, r, mug):
    """Implement alternative drag correlations"""
```

#### Breakup Criteria
**Location:** Lines 817-820
```python
# Add secondary breakup criteria
if (custom_criterion):
    # Fragmentation logic
```

## Output Analysis

### Generated Files

1. **PDF Report:** `Evolution_Moments_MTE_2D_Fragmentation.pdf`
   - Initial conditions summary
   - Final statistics comparison (CQMOM vs MC)
   - Moment evolution plots
   - Size distribution histograms

2. **JSON Data:** `radii_timeseries.json`
   - Time-resolved droplet size data
   - Useful for post-processing and visualization of the droplets radius
   - **Required for GIF animation** (see Visualization section)

3. **Plot_only_CQMOM:**
   - Resolve moments only with CQMOM and visualize directly
   - Useful for quick simulation and very large distribution

4. **Animated GIF:** `droplet_radii_evolution.gif` (optional)
   - Dynamic visualization of droplet size distribution evolution
   - Generated from JSON data using `Generate_gif_radius_evolution.py`

### Key Metrics

| Metric | CQMOM | Monte Carlo |
|--------|-------|-------------|
| **Mean radius** | From M₁₀/M₀₀ | Direct averaging |
| **Mean velocity** | From M₀₁/M₀₀ | Direct averaging |
| **Number of droplets** | M₀₀ | Particle count |
| **Size distribution** | Not available | Full histogram |

### Understanding Moments (CQMOM)

The moment-based approach uses statistical moments to represent the droplet population:

| Moment | Mathematical Definition | Physical Meaning |
|--------|------------------------|------------------|
| **M₀₀** | Σ wₖ | **Total number of droplets** in the system |
| **M₁₀** | Σ wₖ × rₖ | **Weighted sum of radii** (not mean radius directly) |
| **M₀₁** | Σ wₖ × uₖ | **Weighted sum of velocities** (not mean velocity directly) |
| **M₁₁** | Σ wₖ × rₖ × uₖ | **Mixed moment** representing radius-velocity covariance |

Where:
- `wₖ` = weight of droplet k
- `rₖ` = radius of droplet k  
- `uₖ` = velocity of droplet k

#### Derived Quantities:
```python
# Mean values from moments
mean_radius = M₁₀ / M₀₀     # Average droplet radius
mean_velocity = M₀₁ / M₀₀   # Average droplet velocity

# Covariance information
# M₁₁ provides insight into radius-velocity correlation in the population
```

**Important:** CQMOM tracks these moments directly through transport equations and source term, while Monte Carlo calculates them from individual particle data.

## Performance Optimization

### Computational Scaling

- **Monte Carlo:** O(N(T) × T) where N = particles, T = time steps with N growing over time ! This leads to resolving the equations each time N changes at each frame -> Very costly !
- **CQMOM:** O(M² × T) where M = moments (typically M=4) -> Not costly at all

### Parallelization Settings

```python
# Adjust based on your system (line 44)
max_workers = 4  # Recommended: CPU cores - 1
```

### Memory Considerations

```python
# Reduce for memory-limited systems
max_nb_fragmentation = 100  # Default: 250
```

## Validation and Testing

### Recommended Validation Steps

1. **Conservation checks:**
   ```python
   # Volume conservation
   initial_volume = np.sum([(4/3)*np.pi*r**3 for r in initial_radii])
   final_volume = np.sum([(4/3)*np.pi*r**3 for r in final_radii])
   ```

2. **Moment consistency:**
   - Compare CQMOM M₀₀ with MC particle count
   - Verify M₁₀/M₀₀ ≈ MC mean radius

3. **Parameter sensitivity:**
   - Test different Weber numbers and be wary of the Ohnesorge number influence
   - Vary initial conditions
   - Check time step 

## Visualization Functions

### Built-in Plotting

- `plot_results()`: Complete PDF report
- `plot_only_CQMOM()`: CQMOM-only visualization

### Advanced Visualization: Animated GIF Generation

**File:** `Generate_gif_radius_evolution.py`

**Prerequisites:** ⚠️ **Monte Carlo simulation must have been run first** to generate `radii_timeseries.json`

This script creates an animated GIF showing the temporal evolution of droplet size distributions with intelligent features:

#### Usage:
```python
# After running MC simulation
python Generate_gif_radius_evolution.py
```

#### Generated Output:
- **File:** `droplet_radii_evolution.gif`
- **Duration:** Full simulation timeline at 12 fps here


## Troubleshooting

### Common Issues

1. **BrokenProcessPool (Notebook):**
   - **Solution:** Use ThreadPoolExecutor (already implemented)
   - **Alternative:** Run the Python script version

2. **a[k - 1] invalid values or divide by zero from Quadrature/CQMOM.py:**
   - **Symptom:** a[k - 1] invalid values or divide by zero
   - **Solution:** Ignore it. It just happen at the beginning of a simulation and does not have any influence on it.

3. **Memory Issues:**
   - Reduce `max_nb_fragmentation`
   - Decrease `num_particles`
   - Use shorter `simulation_time` by increasing the `time_step` or by reducing the `num_loops`

4. **Simulation runs forever or one branch is remaining but can't be completed:**
   - **Parallelisation problem:** Sometimes, as parallelisation is not always working correctly with windows, a process can not be stopped and so, the branch connected to the process runs forever, even the simulation of the branch has finished.
   - **What to do:** In order to not lose all the results of the simulation, you can press one time in the terminal "Ctrl + C". That will indicate to the code to force the stoppage of the parallelisation process only, and not the entire code. Therefore, it will keep all the results of the branches and you will still have access to X numbers of droplets simulated.
   - **Don't work?:** Press "Ctrl + C" a couple of more times with at least 2 seconds between each press. If nothing happened, you have to kill the terminal unfortunately.

5. **GIF Generation Issues:**
   - **Missing JSON file:** Ensure Monte Carlo simulation completed successfully
   - **Empty GIF:** Check that `radii_timeseries.json` contains data
   - **Takes too much time:** Large simulations may require chunked processing or reduced frame rate but with patience, you will obtain the gif

6. **Numerical Robustness Issues:**
   
   ⚠️ **Critical:** Boundary values and numerical limits can cause simulation instabilities
   
   **Example: Zero or near-zero characteristic times**
   ```python
   # In drdt_with_fragmentation_shrinkage() and calculate_fragmentation_source_rate()
   taubag = np.pi * ((rhol*r**3)/(2*sigmal))**(1/2)
   tausheer = B2RD * r * (rhol/rhog)**(1/2) / abs(u - ug)
   
   # Issues: Division by zero, extremely small values
   ```
   
   **Solutions implemented:**
   - **Minimum thresholds:** `if taubag < 1e-9: taubag = -1` (lines 797-800)
   - **Velocity protection:** `if abs(u - ug) < 1e-10: continue` (lines 360, 464, 792)
   - **NaN/Inf checks:** Throughout fragmentation functions
   
   **Example: Unphysical droplet sizes**
   ```python
   # Very small droplets from numerical errors can destabilize simulations
   # Current protection (line 777):
   if vol_candidate > (4/3)*np.pi*(0.05*r0)**3:  # Minimum 5% of initial radius
   ```
   
   **Summary**
   - **Symptom:** Sudden appearance of many extremely small droplet, brief peak in the moments, jump of the moments, strong and strange decrease or increase of moments...
   - **Cause:** Numerical errors in boundary calculations propagating
   - **Solution:** Implement size-based fragmentation cutoffs, if statement or try, add values threshold....
   

### Debugging Tips

```python
# Add debug prints in fragmentation loop
print(f"Frame {frame}: Active particles, radius, moments, fragmentation, new child droplets formed...")
print(f"Weber = {We_current_drop:.2f}, Critical = {Wecrit_current_drop:.2f}")
```


