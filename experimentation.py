import torch 
import numpy as np

def conjugate_gradient_hessian_vp(A, gradient):
    """
    trying to solve the linear system hessian (x) = gradient using hessian vector product
    with conjugate gradient 
    """
    tolerance = 10**-3

    x_i = torch.rand(gradient.shape[0])
    #_, hessian_times_x_0 = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x_i)

    r_i = gradient - (A @ x_i )
    p_i = r_i

    #print(torch.norm(r_i))
    if torch.norm(r_i).item() <= tolerance:
        return x_i
    
    num_iter = 30
    for i in range(num_iter): # run for a certain number of iterations 
        #print(hessian_times_p_i)
        alpha_i = (r_i.T @ r_i)/(p_i.T @ A @ p_i)
        x_i = x_i + (alpha_i * p_i)
        
        print(torch.norm(r_i).item())
        r_prev = r_i 
        r_i = r_i - (alpha_i * (A @ p_i))
        if torch.norm(r_i).item() <= tolerance:
            return x_i 
        
        beta_i = (r_i.T @ r_i)/(r_prev.T @ r_prev)
        p_i = r_i + (beta_i * p_i)

    return x_i

A = torch.tensor([[2., 3., 0.], [3., 1., 2.], [0., 2., 1.]])
b = torch.rand(3)
print(torch.inverse(A) @ b)
print(conjugate_gradient_hessian_vp(A, b))