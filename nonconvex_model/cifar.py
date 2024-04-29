import torch 
import torchvision
import torchvision.transforms as transforms 


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


class Model:

    def __init__(self):
        pass 


if __name__ == "__main__":
    train_data_loader, test_data_loader = create_train_test_dataloaders(32)
    print(train_data_loader)
    print(test_data_loader)







