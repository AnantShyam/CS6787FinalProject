import numpy as np 
import torch


def biconjugate_gradient_stable_hessian_vp(A, b):
    num_iter = 6 
    
    x = [None] * num_iter 
    r = [None] * num_iter
    r_hat = [None] * num_iter
    rho = [None] * num_iter
    p = [None] * num_iter

    # initialize parameters
    x[0] = torch.rand(b.shape[0])
    r[0] = b - (A @ x[0])
    r_hat[0] = torch.clone(r[0])

    rho[0] = (r_hat[0].T @ r[0])
    p[0] = r[0]

    for i in range(1, num_iter):
        v = A @ p[i - 1]
        alpha = rho[i - 1]/(r_hat[0].T @ v)
        h = x[i - 1] + (alpha * p[i - 1])
        s = r[i - 1] - (alpha * v)

        if (torch.norm(b - (A @ x[i - 1]))) <= 10**-3:
            return x[i - 1]

        t = A @ s 
        omega = (t.T @ s)/(t.T @ t)
        x[i] = h + (omega * s)
        r[i] = s - (omega * t)
        rho[i] = (r_hat[0].T @ r[i])

        beta = (rho[i]/rho[i - 1]) * (alpha/omega)
        p[i] = r[i] + (beta * (p[i - 1] - (omega * v)))
    
    return x[-1]


def biconjugate_gradient_stable(A, b):
    num_iter = 6 
    
    x = [None] * num_iter 
    r = [None] * num_iter
    r_hat = [None] * num_iter
    rho = [None] * num_iter
    p = [None] * num_iter

    # initialize parameters
    x[0] = torch.rand(b.shape[0])
    r[0] = b - (A @ x[0])
    r_hat[0] = torch.clone(r[0])

    rho[0] = (r_hat[0].T @ r[0])
    p[0] = r[0]

    for i in range(1, num_iter):
        v = A @ p[i - 1]
        alpha = rho[i - 1]/(r_hat[0].T @ v)
        h = x[i - 1] + (alpha * p[i - 1])
        s = r[i - 1] - (alpha * v)

        if (torch.norm(b - (A @ x[i - 1]))) <= 10**-3:
            return x[i - 1]

        t = A @ s 
        omega = (t.T @ s)/(t.T @ t)
        x[i] = h + (omega * s)
        r[i] = s - (omega * t)
        rho[i] = (r_hat[0].T @ r[i])

        beta = (rho[i]/rho[i - 1]) * (alpha/omega)
        p[i] = r[i] + (beta * (p[i - 1] - (omega * v)))
    
    return x[-1]



A = torch.rand(10, 10)
b = torch.rand(10)
print(torch.inverse(A) @ b)
print(biconjugate_gradient_stable(A, b))









