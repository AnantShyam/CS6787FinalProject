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
        names = [name for name, _ in model.state_dict().items()]
        for i in range(len(list(model.parameters()))):
            if i == 3:
                weight_vector = list(model.parameters())[i]
                original_shape = weight_vector.shape
                weight_vector = weight_vector.flatten()
                weight_vector = weight_vector.reshape(len(weight_vector), 1)

                gradient = grads[i]
                eigenvalues, eigenvectors = hessian_class.eigenvalues(maxIter=1, top_n=1)
                eigenvectors = eigenvectors[0]

                eigenvalue = eigenvalues[0]
                eigenvector = eigenvectors[i].flatten()
                eigenvector = eigenvector.reshape(len(eigenvector), 1)
                
                gradient = gradient.flatten()
                gradient = gradient.reshape(len(gradient), 1)

                # approximate hessian
                print(eigenvector.shape)
                print(eigenvector.T.shape)
                hessian_matrix_first_matrix = (eigenvalue) * (eigenvector @ eigenvector.T)
                
                tau = 10**-5
                hessian_matrix_first_matrix = hessian_matrix_first_matrix + (tau*torch.eye(len(hessian_matrix_first_matrix)))
                #print(hessian_matrix_first_matrix.shape)
                hessian_matrix_first_matrix_inverse = torch.inverse(hessian_matrix_first_matrix)

                alpha = 0.01
                
                # gradient = gradient.flatten()
                # gradient = gradient.reshape(len(gradient), 1)

                new_first_weight_vector = weight_vector - (alpha * (hessian_matrix_first_matrix_inverse @ gradient))
                new_first_layer_weight_vector = new_first_weight_vector.reshape(original_shape)
                #print(model.state_dict()['conv1.weight'])

                #x = copy.deepcopy(model.state_dict()['conv1.weight'])
                model.state_dict()[names[i]].data += new_first_layer_weight_vector

                #y = copy.deepcopy(model.state_dict()['conv1.weight'])
                # print(model.state_dict()['conv1.weight'].data[0,0])
                # print(x.data[0,0])
                #print(y.data[0,0] - x.data[0,0])
                print('done')

    return model
    


if __name__ == "__main__":
    train_data_loader, test_data_loader = cifar.create_train_test_dataloaders(2000)
    initial_model = cifar_model.CIFAR10Net()
    #train_newton_method(initial_model, train_data_loader, 1)
    trained_model = newton_method(initial_model, train_data_loader, 1)