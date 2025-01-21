import numpy as np
import matplotlib.pyplot as plt

def fractal_dimension_3d(grid):
    """
    Calculate the fractal dimension of a 3D object using the box-counting method.
    Args:
        grid: A 3D numpy array where 1 indicates the object and 0 is empty space.
    Returns:
        fractal_dimension: The fractal dimension of the object.
    """
    # Ensure the grid is a binary array
    grid = (grid > 0)
    
    # Get the grid size
    size = grid.shape[0]  # Assume the grid is a cube (NxNxN)
    assert grid.shape[0] == grid.shape[1] == grid.shape[2], "Grid must be a cube!"
    
    # Define box sizes (logarithmic scale)
    scales = np.floor(np.logspace(0, np.log2(size), num=10, base=2, endpoint=False)).astype(int)
    scales = scales[scales > 1]  # Exclude box size 1 (too small)
    scales = [scale for scale in scales if size % scale == 0]  # Keep only divisible scales

    Ns = []  # Store the number of boxes at each scale
    
    for scale in scales:
            # Divide the grid into boxes of size (scale x scale x scale)
        sub_boxes = (
            grid.reshape(size // scale, scale,
                         size // scale, scale,
                         size // scale, scale)
                .max(axis=(1, 3, 5))
        )
        
        # Count non-empty boxes
        Ns.append(np.sum(sub_boxes > 0))
    
    # Debugging: Print scales and Ns
    print(f"Scales: {scales}")
    print(f"Ns: {Ns}")
    
    # Check if enough points exist for a reliable fit
    if len(scales) < 2:
        raise ValueError("Not enough valid scales for a reliable fit. Increase the grid size or refine scales.")
    
    # Compute the fractal dimension (log-log fit)
    log_scales = np.log(scales)
    log_Ns = np.log(Ns)
    coeffs = np.polyfit(log_scales, log_Ns, 1)
    
    # Fractal dimension is the negative slope
    fractal_dimension = -coeffs[0]
    
    # Plot the results
    plt.figure(figsize=(6, 4))
    plt.plot(log_scales, log_Ns, 'o', mfc='none', label='Data')
    plt.plot(log_scales, np.polyval(coeffs, log_scales), label=f'Fit (D = {fractal_dimension:.3f})')
    plt.xlabel('log(Scale)')
    plt.ylabel('log(Number of Boxes)')
    plt.legend()
    plt.title('3D Fractal Dimension')
    plt.grid()
    plt.show()
    
    return fractal_dimension

# Example: Create a 3D Sierpinski-like cube
def sierpinski_carpet_3d(n, size):
    """Create a 3D Sierpinski carpet fractal."""
    grid = np.ones((size, size, size), dtype=bool)
    for _ in range(n):
        step = size // 3
        grid[step:2*step, step:2*step, :] = 0
        grid[step:2*step, :, step:2*step] = 0
        grid[:, step:2*step, step:2*step] = 0
        size //= 3
    return grid

# Generate a 3D Sierpinski fractal grid
size = 243  # Larger size allows more valid scales
grid = sierpinski_carpet_3d(5, size)  # Increase iterations for finer detail

# Compute its fractal dimension
try:
    fractal_dim = fractal_dimension_3d(grid)
    print(f"Fractal Dimension: {fractal_dim:.3f}")
except ValueError as e:
    print("Error:", e)
