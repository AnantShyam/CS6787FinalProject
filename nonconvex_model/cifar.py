import torch 
import torchvision
from torch import optim
import torchvision.transforms as transforms 
import cifar_model
import time
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def train_model_gradient_descent(model, train_data_loader, num_epochs=10):
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    
    loss_values = []
    mean_accuracies = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0

        accuracies = []
        for inputs, labels in train_data_loader:

            outputs = model(inputs)
            loss = l2_regularized_cross_entropy_loss(outputs, labels, model, 0)
            total_loss += loss.item()

            acc = accuracy(outputs, labels)
            accuracies.append(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        mean_accuracy = torch.mean(torch.tensor(accuracies)).item()
        print(mean_accuracy)
        mean_accuracies.append(mean_accuracy)

        loss_values.append(total_loss)

    return model, loss_values, mean_accuracies
            
    
def plot_values(x_vals, y_vals, x_axis_title, y_axis_title, file_path_name):
    plt.plot(x_vals, y_vals)
    plt.xlabel(f'{x_axis_title}')
    plt.ylabel(f'{y_axis_title}')
    plt.savefig(file_path_name)

# w vector = w .parameters
# torch.nn.utils.parameters_to_vector()
if __name__ == "__main__":
    train_data_loader, test_data_loader = create_train_test_dataloaders(32)
    initial_model = cifar_model.CIFAR10Net()
    torch.save(initial_model, 'model_weights/initial_model_weights.pt')
    num_epochs = 10
    trained_model, train_loss_values, train_mean_accuracies = train_model_gradient_descent(initial_model, train_data_loader, num_epochs)
    
    torch.save(trained_model, 'model_weights/trained_model_weights.pt')
    torch.save(torch.tensor(train_loss_values), 'model_statistics/train_loss_values.pt')
    torch.save(torch.tensor(train_mean_accuracies), 'model_statistics/train_mean_accuracies.pt')
    
    plot_values([i for i in range(1, num_epochs + 1)], train_loss_values, 'Epoch Number', 'Train Loss Values', 
    'nonconvex_model_plots/train_loss.png')

    plot_values([i for i in range(1, num_epochs + 1)], train_mean_accuracies, 'Epoch Number', 'Train Mean Accuracy', 
    'nonconvex_model_plots/train_mean_accuracy.png')

    # torch.save(trained_model, 'model_weights/trained_model_weights.pt')
    # torch.save(torch.tensor(train_loss_values), 'model_statistics/train_loss_values.pt')
    # torch.save(torch.tensor(train_accuracies), 'model_statistics/train_accuracies.pt')
    
    







