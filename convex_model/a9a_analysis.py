import numpy as np 
import torch
import sklearn 
import helpers
from tqdm import tqdm
import matplotlib.pyplot as plt
import hessian
import scipy
import time 
import os
            
class A9A_Analysis:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X = X_train    
        self.y = y_train 
        self.X_test = X_test 
        self.y_test = y_test

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
            print(curr_loss)
            if abs(curr_loss - loss_values[-1]) <= tolerance:
                break
            else:
                loss_values.append(curr_loss)

        return w, loss_values[1:]
        
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

        r[0] = b - torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x[0])[1]
        r_hat[0] = torch.clone(r[0])

        rho[0] = (r_hat[0].T @ r[0])
        p[0] = r[0]


        while True:

            v = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, p[-1])[1]
            alpha = rho[-1]/(r_hat[0].T @ v)
            h = x[-1] + (alpha * p[-1])
            s = r[-1] - (alpha * v)

            dist_to_solution = b - torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, x[-1])[1]
            if (torch.norm(dist_to_solution)) <= tol:
                return x[-1], num_iter

            t = torch.autograd.functional.hvp(self.l2_regularized_logistic_regression_loss, w, s)[1]
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
            gradient = torch.autograd.functional.jacobian(self.l2_regularized_logistic_regression_loss, w) 
            hessian = torch.func.hessian(self.l2_regularized_logistic_regression_loss)(w)
            
            # add regularization to hessian to ensure it is positive definite
            hessian = hessian + (0.01 * torch.eye(len(hessian)))
            
            # solve the linear system with stable biconjugate gradient method using hessian vector products
            update, _ = self.biconjugate_gradient_stable(w, gradient, 10000)
            w = w - (0.1 * update) 
            loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            loss_values[epoch + 1] = loss_val
            #print(loss_val)

        return w, loss_values



    def measure_wall_clock_time(self, optimization_method, num_epochs):
        # start_time = time.time()
        # _ = self.newton_method_exact(num_epochs) 
        # end = time.time()

        # print(end - start_time)

        start = time.time()
        _ = optimization_method(num_epochs)
        end_time = time.time()
        print(end_time - start)

        pass 


    def test_model(self, weight_vector):
        num_correct = 0

        n = len(self.y_test)
        for i in range(n):
            training_example, label = self.X_test[:, i], self.y_test[i]
            prediction = torch.sign(weight_vector.T @ training_example).sign()
            # print(prediction)
            # print(label)
            # print('----')
            if prediction == label:
                num_correct += 1
        
        return float(num_correct)/float(n)


    def plot_suboptimality(self, filename, newton_method):
        # difference between Gradient Descent approximately converged loss and Newton's Method Loss over iterations 
        
        #gradient_descent_loss = self.gradient_descent()
        #torch.save(torch.tensor(gradient_descent_loss), 'model_training_information/gradient_descent_loss.pt')
        gradient_descent_loss = torch.load('../model_training_information/gradient_descent_loss.pt')
        final_converged_loss = gradient_descent_loss[-1]
        
        _, newton_method_loss_vals = newton_method(15)
         
        # torch.save(torch.tensor(list(newton_method_loss_vals.values())), 'model_training_information/biconjugate_loss.pt')
        torch.save(torch.tensor(list(newton_method_loss_vals.values())), '../model_training_information/newton_method_loss_a9a.pt')
        
        loss_differences = {i: abs(newton_method_loss_vals[i] - final_converged_loss) 
                                for i in range(1, len(newton_method_loss_vals))}

        helpers.plot_curve(list(loss_differences.keys()), list(loss_differences.values()), 
        'Number of Epochs', 'Suboptimality', filename)




if __name__ == "__main__":
    a9a_dataset_train, labels_train = helpers.read_a9a_dataset('../data/a9a_train.txt')
    a9a_dataset_test, labels_test = helpers.read_a9a_dataset('../data/a9a_test.txt')


    a9a = A9A_Analysis(a9a_dataset_train, labels_train, a9a_dataset_test, labels_test)
    #a9a.plot_suboptimality('biconjugate_gradient_suboptimality.png', a9a.sketch_newton_method)
    #a9a.plot_suboptimality('newton_method_suboptimality_a9a.png', a9a.newton_method_exact)
    
    a9a.measure_wall_clock_time(a9a.sketch_newton_method, 15)
    a9a.measure_wall_clock_time(a9a.newton_method_exact, 15)
    #w, _ = a9a.gradient_descent()
    #print(a9a.test_model(w))

    #w, _ = a9a.sketch_newton_method(8)
    #print("-----")
    #print(a9a.test_model(w))
    # print(a9a.sketch_newton_method(8))
    # print(a9a.X.shape)
    # a9a.plot_suboptimality()

    
    # print(loss_values)
    # plt.plot([key for key in loss_values], [loss_values[key] for key in loss_values])
    # plt.xlabel("Epoch Number")
    # plt.ylabel("Loss")
    # plt.title("Loss vs Epoch Number (Standard Newton's Method on a9a dataset)")
    # plt.show()
