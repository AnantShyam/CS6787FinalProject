import torch 
import numpy as np


def conjugate_gradient_hessian_vp(self, gradient, w):
    """
    trying to solve the linear system hessian (x) = gradient using hessian vector product
    with conjugate gradient 
    """
    l2_logistic_loss = lambda x: self.l2_regularized_logistic_regression_loss(x)
    tolerance = 10**-3

    x_i = torch.rand(gradient.shape[0])
    _, hessian_times_x_0 = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x_i)

    r_i = gradient - hessian_times_x_0
    p_i = r_i

    #print(torch.norm(r_i))
    if torch.norm(r_i).item() <= tolerance:
        return x_i
    
    num_iter = 15
    for i in range(num_iter): # run for a certain number of iterations 
        _, hessian_times_p_i = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x_i)
        #print(hessian_times_p_i)
        alpha_i = (r_i.T @ r_i)/(p_i.T @ hessian_times_p_i)
        x_i = x_i + (alpha_i * p_i)
        
        r_prev = r_i 
        r_i = r_i - (alpha_i * hessian_times_p_i)
        if torch.norm(r_i).item() <= tolerance:
            return x_i 
        
        beta_i = (r_i.T @ r_i)/(r_prev.T @ r_prev)
        p_i = r_i + (beta_i * p_i)

    return x_i


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



def newton_method_svd_low_rank(self, num_epochs):
        """
        Standard implementation of Newton's Method
        Using L2-regularized logistic regression (trying to get a baseline)
        """
        w = torch.rand(self.X.shape[0])

        loss_values = {}
        num_epochs = 8

        for epoch in tqdm(range(num_epochs)):
            
            # compute the gradient 
            gradient = torch.autograd.functional.jacobian(self.l2_regularized_logistic_regression_loss, w)

            # compute the Hessian
            hessian_matrix = torch.func.hessian(self.l2_regularized_logistic_regression_loss)(w)
            hessian_num_rows, hessian_num_columns = hessian_matrix.shape
            
            # do rank k approximation with SVD 
            U, S, V = torch.svd(hessian_matrix)
            k = int(hessian_num_columns/2)
            U_k, S_k, V_k = U[:, 0: k], torch.diag(S[0: k]), V[:, 0: k]
            low_rank_hessian = (U_k @ S_k @ V_k.T)
            
            # add regularization to make hessian positive definite
            tau = 0.001
            low_rank_hessian = low_rank_hessian + (tau * torch.eye(hessian_num_rows))

            # solve linear system with conjugate gradient 
            update = torch.from_numpy(scipy.sparse.linalg.cg(low_rank_hessian.numpy(), gradient.numpy())[0])
            w = w - (0.001 * update)
            loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            loss_values[epoch + 1] = loss_val

        return w, loss_values
        

A = torch.tensor([[2., 3., 0.], [3., 1., 2.], [0., 2., 1.]])
b = torch.rand(3)
print(torch.inverse(A) @ b)
print(conjugate_gradient_hessian_vp(A, b))