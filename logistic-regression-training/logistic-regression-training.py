import numpy as np

def _sigmoid(z):
    # Numerically stable sigmoid
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    X: shape (N, D)
    y: shape (N,)
    lr: learning rate
    steps: number of iterations
    
    Returns:
    w: shape (D,)
    b: float
    """
    
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # Forward pass
        z = np.dot(X, w) + b          # (N,)
        p = _sigmoid(z)               # (N,)
        
        # Compute gradients
        dw = (1 / N) * np.dot(X.T, (p - y))   # (D,)
        db = (1 / N) * np.sum(p - y)          # scalar
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
    
    return w, b