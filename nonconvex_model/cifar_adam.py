import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from cifar10_loader import CIFAR10Dataset
import torch.nn as nn
import torch.optim as optim

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.initialize_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

def train_model(model, trainloader, num_epochs=10, use_adam=True):
    if use_adam:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_batches = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            total_accuracy += accuracy(outputs, labels)
            total_batches += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / total_batches
        avg_accuracy = total_accuracy / total_batches
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1}: Loss = {avg_loss:.2f}, Accuracy = {avg_accuracy:.2f}%, Time = {elapsed_time:.2f} seconds')

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_path = '/Users/cba/Desktop/cs_6787/final_project/CS6787FinalProject/data/cifar-10-batches-py'
    train_dataset = CIFAR10Dataset(root_dir=dataset_path, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    model = CIFAR10Net().to('cpu')
    train_model(model, train_loader, num_epochs=20, use_adam=True)

if __name__ == "__main__":
    main()
