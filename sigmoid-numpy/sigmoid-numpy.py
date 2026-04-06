import numpy as np

def sigmoid(x):
    # Convert input to NumPy array (handles scalars, lists, arrays)
    x = np.array(x, dtype=float)
    
    # Apply sigmoid function
    return 1 / (1 + np.exp(-x))