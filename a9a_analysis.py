import numpy as np 
import torch
import sklearn 
import helpers
from tqdm import tqdm
import matplotlib.pyplot as plt
import hessian
import scipy
            
class A9A_Analysis:

    def __init__(self, X, y):
        self.X = X 
        self.y = y

    def l2_regularized_logistic_regression_loss(self,w):
        loss = 0
        n = len(self.y)
        for i in range(n):
            training_example, label = self.X[:, i], self.y[i]
            
            concat_tensor = torch.tensor([0, (-w.T @ training_example) * label])
            exp_term = torch.logsumexp(concat_tensor, 0)

            loss = loss + exp_term

        regularization_constant = 0.05
        regularization_term = (regularization_constant * torch.pow(torch.norm(w), 2)/2.0)
        total_loss = (loss/n) + regularization_term
        return total_loss

    # H = U (Sigma) V^T 
    # H = A B, A = first column of U, B = first column of V^T

    def gradient_descent(self):
        d, n = self.X.shape
        w = torch.rand(d)

        # compute the largest eigenvalue of Hessian at w = 0, hessian = 124 x 124
        # H = 1/n  X^T P X, P = 0.25 I when w = 0
        hessian_w_equals_0 = (1/n) * (self.X @ (0.25 * torch.eye(n)) @ self.X.T) 

        # assuming all eigenvalues are essentially real
        eigenvalues_hessian_w_equals_0 = torch.view_as_real(torch.linalg.eig(hessian_w_equals_0)[0])
    
        largest_eigenvalue = torch.max(eigenvalues_hessian_w_equals_0)

        alpha = 2/largest_eigenvalue
        loss_values = [float('inf')]

        tolerance = 10**(-16)
        while True:
            gradient = torch.autograd.functional.jacobian(self.l2_regularized_logistic_regression_loss, w)
            w = w - (alpha * gradient)
            curr_loss = self.l2_regularized_logistic_regression_loss(w).item()
            if abs(curr_loss - loss_values[-1]) <= tolerance:
                break
            else:
                loss_values.append(curr_loss)

        return loss_values[1:]
        
    def newton_method_exact(self, num_epochs):
        w = torch.rand(self.X.shape[0])
        loss_values = {}

        for epoch in tqdm(range(num_epochs)):

            # compute the gradient 
            gradient = torch.autograd.functional.jacobian(self.l2_regularized_logistic_regression_loss, w)

            # compute the hessian
            hessian_matrix = torch.func.hessian(self.l2_regularized_logistic_regression_loss)(w)
            hessian_num_rows, hessian_num_columns = hessian_matrix.shape

            # update weights 
            hessian_inverse = torch.inverse(hessian_matrix)
            w = w - (0.5 * (hessian_inverse @ gradient))

            loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            loss_values[epoch + 1] = loss_val
        
        return w, loss_values

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


    def conjugate_gradient_hessian_vp(self, gradient, w):
        """
        trying to solve the linear system hessian (x) = gradient using hessian vector product
        with conjugate gradient 
        """
        l2_logistic_loss = lambda x: self.l2_regularized_logistic_regression_loss(x)
        tolerance = 10**-3

        x_i = torch.rand(gradient.shape[0])
        _, hessian_times_x_0 = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x_i)

        # print(gradient.shape)
        # print(hessian_times_x_0.shape)
        r_i = gradient - hessian_times_x_0
        p_i = r_i

        if torch.norm(r_i) <= tolerance:
            return x_i
        
        num_iter = 2
        for i in range(num_iter): # run for a certain number of iterations 
            _, hessian_times_p_i = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x_i)
            #print(hessian_times_p_i)
            alpha_i = (r_i.T @ r_i)/(p_i.T @ hessian_times_p_i)
            #print(alpha_i)
            #print(p_i)
            x_i = x_i + (alpha_i * p_i)
            

            r_prev = r_i 
            r_i = r_i - (alpha_i * hessian_times_p_i)
            if torch.norm(r_i) <= tolerance:
                return x_i 
            
            beta_i = (r_i.T @ r_i)/(r_prev.T @ r_prev)
            p_i = r_i + (beta_i * p_i)

        return x_i



    def sketch_newton_method(self, num_epochs):
        """
        Idea: don't directly compute the Hessian. Solve linear system H p = g with hessian vector products

        let loss function be L 
        Define g(w) = v.T @ grad_L(w)

        grad_g(w) = H v, H = hessian of loss function L

        """
        w = torch.rand(self.X.shape[0])
         
        for _ in tqdm(range(num_epochs)):

            # compute the gradient 
            gradient = torch.autograd.functional.jacobian(self.l2_regularized_logistic_regression_loss, w) 

            # solve the linear system with conjugate gradient using hessian vector products
            update = self.conjugate_gradient_hessian_vp(gradient, w)
            w = w - (0.1 * update) 
            loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            print(loss_val)

        return w 




    def plot_suboptimality(self):
        # difference between Gradient Descent approximately converged loss and Newton's Method Loss over iterations 
        #final_converged_loss = self.gradient_descent()[-1]
        #final_converged_loss=0
        _, newton_method_loss_vals = self.newton_method_exact(8)
        
        loss_differences = {i: abs(newton_method_loss_vals[i] - final_converged_loss) 
                                for i in range(1, len(newton_method_loss_vals))}

        helpers.plot_curve(list(loss_differences.keys()), list(loss_differences.values()), 
        'Number of Epochs', 'Suboptimality', 'a9a_suboptimality_approx_hessian.png')
        


if __name__ == "__main__":
    a9a_dataset, labels = helpers.read_a9a_dataset('data/a9a_train.txt')
    a9a = A9A_Analysis(a9a_dataset, labels)

    print(a9a.sketch_newton_method(10))
    # print(a9a.X.shape)
    # a9a.plot_suboptimality()

    
    # print(loss_values)
    # plt.plot([key for key in loss_values], [loss_values[key] for key in loss_values])
    # plt.xlabel("Epoch Number")
    # plt.ylabel("Loss")
    # plt.title("Loss vs Epoch Number (Standard Newton's Method on a9a dataset)")
    # plt.show()
