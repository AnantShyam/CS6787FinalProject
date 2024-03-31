import numpy as np 
import torch
import sklearn 
import helpers
from tqdm import tqdm
import matplotlib.pyplot as plt
            
class NewtonMethod:

    def __init__(self, X, y):
        self.X = X 
        self.y = y

    def l2_regularized_logistic_regression_loss(self,w):
        loss = 0
        n = len(self.y)
        for i in range(n):
            training_example, label = self.X[:, i], self.y[i]
            loss = loss + (torch.log(1 + torch.exp((-w.T @ training_example) * label)))

        regularization_constant = 0.05
        regularization_term = (regularization_constant * torch.pow(torch.norm(w), 2)/2.0)
        total_loss = (loss/n) + regularization_term
        return total_loss


    def newton_method(self):
        """
        Standard implementation of Newton's Method
        Using L2-regularized logistic regression (trying to get a baseline)
        """
        w = torch.rand(self.X.shape[0])

        loss_values = {}

        for epoch in tqdm(range(8)):
            hessian_matrix = torch.func.hessian(self.l2_regularized_logistic_regression_loss)(w)
            gradient = torch.autograd.functional.jacobian(self.l2_regularized_logistic_regression_loss, w)
            hessian_inverse = torch.inverse(hessian_matrix)
            w = w - (0.5 * (hessian_inverse @ gradient))
            loss_values[epoch + 1] = self.l2_regularized_logistic_regression_loss(w).item()

        return w, loss_values



if __name__ == "__main__":
    a9a_dataset, labels = helpers.read_a9a_dataset('data/a9a_train.txt')
    newton_m = NewtonMethod(a9a_dataset, labels)
    new_w, loss_values  = newton_m.newton_method()
    
    print(loss_values)
    plt.plot([key for key in loss_values], [loss_values[key] for key in loss_values])
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch Number (Standard Newton's Method on a9a dataset)")
    plt.show()
