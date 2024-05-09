import torch 
import cifar 
import cifar_model
from cifar10_loader import CIFAR10Dataset  
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import pyhessian
import copy

# w vector = w .parameters
# torch.nn.utils.parameters_to_vector()

# compute eigendecomposition of hessian 
# take k positive eigenvalues, k <= # num of positive eigenvalues of hessian 

# Hessian = d x d = V D V^T

# d * k k * k k* d
# add some regularization 

# if no positive eigenvalues, take a gradient step and try again

def diagonal_hessian_approximation(model, loss_fn):
    model_parameters = list(model.parameters())



    pass 




def train_newton_method(model, train_data_loader, num_epochs):
    #print(torch.nn.utils.parameters_to_vector(model.parameters()))
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_data_loader:
            outputs = model(inputs)
            #print(outputs)
            model_params = list(model.parameters())
            hessian_class = pyhessian.hessian(model, F.cross_entropy, dataloader=train_data_loader, cuda=False)

            print(hessian_class.eigenvalues(maxIter=1))

            quit()
    return 


def newton_method(model, data_loader, num_epochs):
    
    for inputs, labels in data_loader:
        hessian_class = pyhessian.hessian(model, F.cross_entropy, data=(inputs, labels), cuda=False)
        params, grads = pyhessian.utils.get_params_grad(model)

        # update the first layer's parameters for right now
        first_layer_weight_vector = list(model.parameters())[0]
        first_layer_weight_vector = first_layer_weight_vector.flatten()
        first_layer_weight_vector = first_layer_weight_vector.reshape(864, 1)

        first_layer_gradient = grads[0]
        eigenvalues, eigenvectors = hessian_class.eigenvalues(maxIter=1)
        eigenvectors = eigenvectors[0]

        eigenvalue = eigenvalues[0]
        eigenvector = eigenvectors[0].flatten()
        eigenvector = eigenvector.reshape(864, 1)
        hessian_matrix_first_matrix = (eigenvalue) * (eigenvector @ eigenvector.T)
        hessian_matrix_first_matrix_inverse = torch.inverse(hessian_matrix_first_matrix)

        alpha = 0.01
        
        first_layer_gradient = first_layer_gradient.flatten().reshape(864, 1)

        new_first_weight_vector = first_layer_weight_vector - (alpha * (hessian_matrix_first_matrix_inverse @ first_layer_gradient))
        new_first_layer_weight_vector = new_first_weight_vector.reshape(32, 3, 3, 3)
        #print(model.state_dict()['conv1.weight'])

        x = model.state_dict()['conv1.weight']
        # for name, param in model.state_dict().items():
        #     if name == 'conv1.weight':
        #         transformed_param = new_first_layer_weight_vector
        #         print(transformed_param - param)
        #         param.copy_(transformed_param)
        # model.load_state_dict(model.state_dict())
        state_dict = copy.deepcopy(model.state_dict())
        state_dict['conv1.weight'] = new_first_layer_weight_vector
        model.load_state_dict(state_dict)
        y = model.state_dict()['conv1.weight']
        print(y - x)
        #print(model.state_dict()['conv1.weight'])
        print('done')
    


if __name__ == "__main__":
    train_data_loader, test_data_loader = cifar.create_train_test_dataloaders(32)
    initial_model = cifar_model.CIFAR10Net()
    #train_newton_method(initial_model, train_data_loader, 1)
    newton_method(initial_model, train_data_loader, 1)