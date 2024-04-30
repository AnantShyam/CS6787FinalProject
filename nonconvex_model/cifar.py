import torch 
import torchvision
from torch import optim
import torchvision.transforms as transforms 
import cifar_model
import time
from torch.nn import functional as F
from tqdm import tqdm

def load_data():
    # normalize data 
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # get training and testing data
    training_data = torchvision.datasets.CIFAR10(root='./data', train=True, 
                            download=True, transform=transform)

    testing_data = torchvision.datasets.CIFAR10(root='./data', train= False, 
                download=True, transform=transform)

    return training_data, testing_data


def create_train_test_dataloaders(batch_size):
    training_data, testing_data = load_data()
    train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True)
    return train_data_loader, test_data_loader


def l2_regularized_cross_entropy_loss(outputs, labels, model, reg_lambda=0.05):
    cross_entropy_loss = F.cross_entropy(outputs, labels)
    l2_reg = sum(torch.norm(param)**2 for param in model.parameters())
    return cross_entropy_loss + reg_lambda * l2_reg


def train_model_gradient_descent(model, train_data_loader, num_epochs=10):
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    
    loss_values = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_data_loader:

            outputs = model(inputs)
            loss = l2_regularized_cross_entropy_loss(outputs, labels, model, 0)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_values.append(total_loss)
    
    return model, loss_values
            
    


# w vector = w .parameters
# torch.nn.utils.parameters_to_vector()
if __name__ == "__main__":
    train_data_loader, test_data_loader = create_train_test_dataloaders(32)
    initial_model = cifar_model.CIFAR10Net()
    trained_model = train_model(initial_model, train_data_loader)
    torch.save()
    







