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
        

    def biconjugate_gradient_stable(self, w, b, tol=10**-3):

        # A = hessian 
        num_iter = 1
        
        x = [None]
        r = [None]
        r_hat = [None]
        rho = [None]
        p = [None]

        # initialize parameters
        x[0] = torch.rand(b.shape[0])

        hessian = self.form_hessian(w)

        print(hessian)
        #print(hessian)
        #r[0] = b - torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x[0])[1]
        r[0] = b - (hessian @ x[0])
        r_hat[0] = torch.clone(r[0])

        rho[0] = (r_hat[0].T @ r[0])
        p[0] = r[0]


        while True:

            #v = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, p[-1])[1]
            v = hessian @ p[-1]
            alpha = rho[-1]/(r_hat[0].T @ v)
            h = x[-1] + (alpha * p[-1])
            s = r[-1] - (alpha * v)

            #dist_to_solution = b - torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x[-1])[1]
            dist_to_solution = b - (hessian @ x[-1])
            if (torch.norm(dist_to_solution)) <= tol:
                return x[-1], num_iter

            #t = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, s)[1]
            t = hesian @ s
            omega = (t.T @ s)/(t.T @ t)
            x.append(h + (omega * s))
            r.append(s - (omega * t))
            rho.append(r_hat[0].T @ r[-1])

            beta = (rho[-1]/rho[-2]) * (alpha/omega)
            p.append(r[-1] + (beta * (p[-1] - (omega * v))))

            num_iter += 1
        
        return x[-1], num_iter



    def sketch_newton_method(self, num_epochs):
        """
        Idea: don't directly compute the Hessian. Solve linear system H p = g with hessian vector products

        let loss function be L 
        Define g(w) = v.T @ grad_L(w)

        grad_g(w) = H v, H = hessian of loss function L

        """
        w = torch.rand(self.X.shape[0])
         
        loss_values = {}
        for epoch in tqdm(range(num_epochs)):
            # compute the gradient, hessian
            gradient = (torch.sigmoid(-self.y*((self.X.T)@w))*(-self.y*self.X)).mean(1) + (self.regularization * w)

            g = lambda u: u.T @ gradient

            update, _ = self.biconjugate_gradient_stable(w, gradient, 10000)
            w = w - (0.01 * update) 
            loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            loss_values[epoch + 1] = loss_val
            print(loss_val)
            print(self.test_model(w))

        return w, loss_values
        
A = torch.tensor([[2., 3., 0.], [3., 1., 2.], [0., 2., 1.]])
b = torch.rand(3)
print(torch.inverse(A) @ b)
print(conjugate_gradient_hessian_vp(A, b))