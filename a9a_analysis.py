import numpy as np 
import torch
import sklearn 
import helpers
from tqdm import tqdm
import matplotlib.pyplot as plt
import hessian
import scipy
import os
            
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


    def biconjugate_gradient_stable(self, w, b):

        # A = hessian 
        num_iter = 15
        
        x = [None] * num_iter 
        r = [None] * num_iter
        r_hat = [None] * num_iter
        rho = [None] * num_iter
        p = [None] * num_iter

        # initialize parameters
        x[0] = torch.rand(b.shape[0])

        r[0] = b - torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x[0])[1]
        #r[0] = b - (A @ x[0])
        r_hat[0] = torch.clone(r[0])

        rho[0] = (r_hat[0].T @ r[0])
        p[0] = r[0]

        for i in range(1, num_iter):

            v = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, p[i - 1])[1]
            # v = A @ p[i - 1]
            alpha = rho[i - 1]/(r_hat[0].T @ v)
            h = x[i - 1] + (alpha * p[i - 1])
            s = r[i - 1] - (alpha * v)

            dist_to_solution = b - torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x[i - 1])[1]
            if (torch.norm(dist_to_solution)) <= 10**-3:
                return x[i - 1]

            t = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, s)[1]
            #t = A @ s 
            omega = (t.T @ s)/(t.T @ t)
            x[i] = h + (omega * s)
            r[i] = s - (omega * t)
            rho[i] = (r_hat[0].T @ r[i])

            beta = (rho[i]/rho[i - 1]) * (alpha/omega)
            p[i] = r[i] + (beta * (p[i - 1] - (omega * v)))
        
        return x[-1]


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

            # compute the gradient 
            gradient = torch.autograd.functional.jacobian(self.l2_regularized_logistic_regression_loss, w) 
            hessian = torch.func.hessian(self.l2_regularized_logistic_regression_loss)(w)
            
            hessian = hessian + (0.01 * torch.eye(len(hessian)))
            
            # solve the linear system with conjugate gradient using hessian vector products

            update = self.biconjugate_gradient_stable(w, gradient)
            #update = self.conjugate_gradient_hessian_vp(gradient, w)
            w = w - (0.1 * update) 
            loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            loss_values[epoch + 1] = loss_val
            print(loss_val)

        return w, loss_values


    def plot_suboptimality(self, filename, newton_method):
        # difference between Gradient Descent approximately converged loss and Newton's Method Loss over iterations 
        
        #gradient_descent_loss = self.gradient_descent()
        #torch.save(torch.tensor(gradient_descent_loss), 'model_training_information/gradient_descent_loss.pt')
        
        gradient_descent_loss = torch.load('model_training_information/gradient_descent_loss.pt')
        final_converged_loss = gradient_descent_loss[-1]
        
        #final_converged_loss=0
        
        _, newton_method_loss_vals = newton_method(15)
         
        torch.save(torch.tensor(list(newton_method_loss_vals.values())), 'model_training_information/biconjugate_loss.pt')
        
        loss_differences = {i: abs(newton_method_loss_vals[i] - final_converged_loss) 
                                for i in range(1, len(newton_method_loss_vals))}

        helpers.plot_curve(list(loss_differences.keys()), list(loss_differences.values()), 
        'Number of Epochs', 'Suboptimality', filename)




if __name__ == "__main__":
    a9a_dataset, labels = helpers.read_a9a_dataset('data/a9a_train.txt')
    a9a = A9A_Analysis(a9a_dataset, labels)
    a9a.plot_suboptimality('biconjugate_gradient_suboptimality.png', a9a.sketch_newton_method)


    # print(a9a.sketch_newton_method(8))
    # print(a9a.X.shape)
    # a9a.plot_suboptimality()

    
    # print(loss_values)
    # plt.plot([key for key in loss_values], [loss_values[key] for key in loss_values])
    # plt.xlabel("Epoch Number")
    # plt.ylabel("Loss")
    # plt.title("Loss vs Epoch Number (Standard Newton's Method on a9a dataset)")
    # plt.show()
