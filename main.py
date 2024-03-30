import numpy as np 
import torch
import sklearn 
import helpers
            
def newton_method(X, y):
    """
    Standard implementation of Newton's Method
    Using L2-regularized logistic regression (trying to get a baseline)
    """
    pass


if __name__ == "__main__":
    a9a_dataset, labels = helpers.read_a9a_dataset('data/a9a_train.txt')
    w = torch.ones(124).to(torch.float32)
    loss = helpers.l2_regularized_logistic_regression_loss(w, a9a_dataset, labels)
    print(loss)