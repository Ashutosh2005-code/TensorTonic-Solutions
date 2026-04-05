import numpy as np

def ridge_regression(X, y, lam):
    # Convert to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    # Step 1: Compute X^T X
    XtX = X.T @ X
    
    # Step 2: Add λI (identity matrix)
    d = XtX.shape[0]
    I = np.eye(d)
    regularized_matrix = XtX + lam * I
    
    # Step 3: Compute inverse
    inv_matrix = np.linalg.inv(regularized_matrix)
    
    # Step 4: Compute X^T y
    Xty = X.T @ y
    
    # Step 5: Final weights
    w = inv_matrix @ Xty
    
    return w.tolist()