import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import Counter
import os

def load_radii_data(json_path):
    """
    Loads radius data from the JSON file
    """
    print("Loading data...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # The data is in the 'data' field
    radii_timeseries = data['data']
    print(f"Number of time steps: {len(radii_timeseries)}")
    
    return radii_timeseries

def detect_stabilization_point(radii_timeseries, window_size=20, threshold=0.02):
    """
    Detects the point where the distribution stabilizes by analyzing 
    the variation of statistical moments
    """
    if len(radii_timeseries) < window_size * 2:
        return len(radii_timeseries) // 2
    
    means = []
    stds = []
    
    # Calculate mean and standard deviation for each time step
    for timestep_radii in radii_timeseries:
        if len(timestep_radii) > 0:
            means.append(np.mean(timestep_radii))
            stds.append(np.std(timestep_radii))
        else:
            means.append(0)
            stds.append(0)
    
    # Look for where the variation becomes small
    for i in range(window_size, len(means) - window_size):
        # Relative variation in a sliding window
        window_means = means[i:i+window_size]
        window_stds = stds[i:i+window_size]
        
        if len(window_means) > 1:
            mean_variation = np.std(window_means) / (np.mean(window_means) + 1e-10)
            std_variation = np.std(window_stds) / (np.mean(window_stds) + 1e-10)
            
            # If variations are small, we consider it stabilized
            if mean_variation < threshold and std_variation < threshold:
                print(f"Stabilized distribution detected at time step: {i}")
                return i
    
    # If no stabilization detected, use a default value
    default_point = max(160, len(radii_timeseries) * 2 // 3)
    print(f"No clear stabilization detected, using time step: {default_point}")
    return min(default_point, len(radii_timeseries) - 1)

def calculate_scales(radii_timeseries, stabilization_point):
    """
    Calculates scales before and after stabilization
    """
    # Global scale: from 0 to maximum radius of all time steps
    all_radii = []
    for timestep_radii in radii_timeseries:
        all_radii.extend(timestep_radii)
    
    if all_radii:
        global_min = 0  # Always start at 0
        global_max = max(all_radii)
    else:
        global_min, global_max = 0, 1
    
    # Stabilized scale (for the last time steps) - VERY PRECISE ZOOM
    all_radii_stable = []
    for i in range(stabilization_point, len(radii_timeseries)):
        all_radii_stable.extend(radii_timeseries[i])
    
    if all_radii_stable:
        # Use percentiles to focus on the core of the distribution
        # and avoid outliers
        radii_array = np.array(all_radii_stable)
        
        # Take 98% of the distribution (exclude 1% of extreme values from each side)
        stable_min_percentile = np.percentile(radii_array, 1)
        stable_max_percentile = np.percentile(radii_array, 99)
        
        # Add a very small margin for precise zoom (only 1%)
        range_width = stable_max_percentile - stable_min_percentile
        margin = range_width * 0.01  # Reduction from 5% to 1%
        
        stable_min = max(stable_min_percentile - margin, 0)
        stable_max = stable_max_percentile + margin
        
        print(f"Stabilized distribution - Percentiles 1%-99%: {stable_min_percentile:.6f} to {stable_max_percentile:.6f}")
        print(f"Stabilized distribution range: {range_width:.6f}")
    else:
        stable_min, stable_max = global_min, global_max
    
    print(f"Global scale: {global_min:.6f} to {global_max:.6f}")
    print(f"Zoomed scale (very precise): {stable_min:.6f} to {stable_max:.6f}")
    print(f"Zoom factor: {(global_max - global_min) / (stable_max - stable_min):.1f}x")
    
    return (global_min, global_max), (stable_min, stable_max)

def create_histogram_gif(radii_timeseries, output_path='droplet_radii_evolution.gif'):
    """
    Creates an animated GIF showing the evolution of radius histograms
    with automatic scale adjustment and progressive zoom
    """
    # Detect stabilization point
    stabilization_point = detect_stabilization_point(radii_timeseries)
    
    # Calculate scales
    global_scale, stable_scale = calculate_scales(radii_timeseries, stabilization_point)
    global_min, global_max = global_scale
    stable_min, stable_max = stable_scale
    
    # Parameters for progressive transition
    transition_duration = 2000  # Number of frames for the transition
    transition_start = max(0, stabilization_point - 10)  # Start a bit earlier
    transition_end = stabilization_point + transition_duration - 10
    
    print(f"Progressive transition from frame {transition_start} to frame {transition_end}")
    
    def get_progressive_scale(frame):
        """
        Calculates progressive scale based on frame
        """
        if frame < transition_start:
            # Before transition: global scale
            return global_min, global_max, "Global scale"
        elif frame > transition_end:
            # After transition: zoomed scale
            return stable_min, stable_max, "Zoomed scale"
        else:
            # During transition: progressive interpolation
            progress = (frame - transition_start) / (transition_end - transition_start)
            
            # Smooth transition function (sigmoid)
            smooth_progress = 0.5 * (1 + np.tanh(6 * (progress - 0.5)))
            
            # Linear interpolation between scales
            min_radius = global_min + smooth_progress * (stable_min - global_min)
            max_radius = global_max + smooth_progress * (stable_max - global_max)
            
            # Progress information
            percentage = int(smooth_progress * 100)
            scale_info = f"Zooming in progress ({percentage}%)"
            
            return min_radius, max_radius, scale_info
    
    # Figure configuration
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate(frame):
        ax.clear()
        
        # Current time step data
        current_radii = radii_timeseries[frame]
        
        # Get progressive scale
        min_radius, max_radius, scale_info = get_progressive_scale(frame)
        
        # Create bins for current scale
        n_bins = 50
        bins = np.linspace(min_radius, max_radius, n_bins + 1)
        
        if len(current_radii) > 0:
            # Filter radii in current range
            filtered_radii = [r for r in current_radii if min_radius <= r <= max_radius]
            
            if filtered_radii:
                # Create histogram
                counts, _ = np.histogram(filtered_radii, bins=bins)
                
                # Bin centers for display
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Create bar chart
                bars = ax.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.8, 
                             alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Color bars based on their height
                max_count = max(counts) if max(counts) > 0 else 1
                for bar, count in zip(bars, counts):
                    normalized_height = count / max_count
                    bar.set_color(plt.cm.viridis(normalized_height))
        
        # Chart parameters
        ax.set_xlabel('Droplet Radius', fontsize=12)
        ax.set_ylabel('Number of Droplets', fontsize=12)
        
        # Title with scale information
        title = f'Droplet Radius Distribution - Step {frame + 1}/{len(radii_timeseries)}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # X-axis limits
        ax.set_xlim(min_radius, max_radius)
        
        # Determine Y-limit for current scale
        max_y = 0
        # Calculate over a range of frames for stable Y-limit
        frame_range_start = max(0, frame - 10)
        frame_range_end = min(len(radii_timeseries), frame + 10)
        
        for i in range(frame_range_start, frame_range_end):
            ts_radii = radii_timeseries[i]
            if len(ts_radii) > 0:
                # Use current scale for calculation
                filtered = [r for r in ts_radii if min_radius <= r <= max_radius]
                
                if filtered:
                    hist_counts, _ = np.histogram(filtered, bins=bins)
                    max_y = max(max_y, max(hist_counts))
        
        ax.set_ylim(0, max_y * 1.1 if max_y > 0 else 1)
        
        # Add information
        total_drops = len(current_radii)
        filtered_drops = len([r for r in current_radii if min_radius <= r <= max_radius])
        
        # Calculate current zoom factor
        current_range = max_radius - min_radius
        global_range = global_max - global_min
        zoom_factor = global_range / current_range if current_range > 0 else 1
        
        info_text = f'Total droplets: {total_drops}\n{scale_info}\nVisible droplets: {filtered_drops}\nZoom: {zoom_factor:.1f}x'
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Improve appearance
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(radii_timeseries), 
                                 interval=120, repeat=True, blit=False)
    
    # Save GIF
    print(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer='pillow', fps=12, dpi=100)
    
    print("GIF created successfully!")
    plt.close()

def main():
    # Path to JSON file
    json_path = 'MTE_2D_Droplet_Breakup_python_output/radii_timeseries.json'
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} does not exist.")
        return
    
    try:
        # Load data
        radii_timeseries = load_radii_data(json_path)
        
        # Display some statistics
        print(f"Number of time steps: {len(radii_timeseries)}")
        
        # Statistics for first time steps
        for i in range(min(3, len(radii_timeseries))):
            print(f"Time step {i+1}: {len(radii_timeseries[i])} droplets")
            if len(radii_timeseries[i]) > 0:
                print(f"  Min radius: {min(radii_timeseries[i]):.6f}")
                print(f"  Max radius: {max(radii_timeseries[i]):.6f}")
        
        # Create GIF
        create_histogram_gif(radii_timeseries, 'droplet_radii_evolution.gif')
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
