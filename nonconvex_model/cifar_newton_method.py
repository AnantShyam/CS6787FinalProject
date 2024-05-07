import torch 
import cifar 
import cifar_model
from cifar10_loader import CIFAR10Dataset  
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim

# w vector = w .parameters
# torch.nn.utils.parameters_to_vector()

# compute eigendecomposition of hessian 
# take k positive eigenvalues, k <= # num of positive eigenvalues of hessian 

# Hessian = d x d = V D V^T

# d * k k * k k* d
# add some regularization 

# if no positive eigenvalues, take a gradient step and try again

def train_newton_method(model, train_data_loader, num_epochs):
    #print(torch.nn.utils.parameters_to_vector(model.parameters()))
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for inputs, labels in train_data_loader:
            outputs = model(inputs)
            #print(outputs)
            #print(model.parameters())
            weight_vector = torch.nn.utils.parameters_to_vector(model.parameters())
            # hessian_matrix = torch.autograd.functional.hessian(




            # )


            #loss_fn = lambda w: cifar.l2_regularized_cross_entropy_loss(outputs, labels, w)
            
            print(weight_vector.shape)
            hessian_matrix = torch.func.hessian(
                cifar.l2_regularized_cross_entropy_loss
            )(outputs, labels.float(), model)

            print(hessian_matrix.shape)
            # hessian_matrix = torch.autograd.functional.hessian(
            #     cifar.l2_regularized_cross_entropy_loss, 
            #     (
            #         outputs, 
            #         labels.float(), 
            #         model
            #     )
            # )
            #hessian_matrix = torch.autograd.functional.hessian(loss_fn, torch.nn.utils.parameters_to_vector(model.parameters()))
            #print(hessian_matrix)
            #quit()
    return 



if __name__ == "__main__":
    train_data_loader, test_data_loader = cifar.create_train_test_dataloaders(32)
    initial_model = cifar_model.CIFAR10Net()
    train_newton_method(initial_model, train_data_loader, 1)