import numpy as np


def compute_svd(H):
    # SVD
    U, Sigma, VT = np.linalg.svd(H, full_matrices=True)
    
    # Extract the first columns of U and VT (which is the transpose of V)
    A = U[:, 0].reshape(-1, 1)
    B = VT.T[:, 0].reshape(-1, 1)
    
    # Multiply A by the square root of the largest sv
    A = A * np.sqrt(Sigma[0])
    
    return A, B


m = 3
n = 2
#  random Hessian matrix mn x mn
H = np.random.rand(m*n, m*n) 


A, B = compute_svd(H)


print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)