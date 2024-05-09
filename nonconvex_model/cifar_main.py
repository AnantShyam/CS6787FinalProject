import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from cifar10_loader import CIFAR10Dataset  # Custom CIFAR-10 data loader
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
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Adjusted for 8x8 output size after pooling
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.initialize_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)  # Ensure this matches the expected input size of self.fc1
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

def biconjugate_gradient_stable(A, b):
    num_iter = 6 
    
    x = [None] * num_iter 
    r = [None] * num_iter
    r_hat = [None] * num_iter
    rho = [None] * num_iter
    p = [None] * num_iter

    # initialize parameters
    x[0] = torch.rand(b.shape[0])
    r[0] = b - (A @ x[0])
    r_hat[0] = torch.clone(r[0])

    rho[0] = (r_hat[0].T @ r[0])
    p[0] = r[0]

    for i in range(1, num_iter):
        v = A @ p[i - 1]
        alpha = rho[i - 1]/(r_hat[0].T @ v)
        h = x[i - 1] + (alpha * p[i - 1])
        s = r[i - 1] - (alpha * v)

        if (torch.norm(b - (A @ x[i - 1]))) <= 10**-3:
            return x[i - 1]

        t = A @ s 
        omega = (t.T @ s)/(t.T @ t)
        x[i] = h + (omega * s)
        r[i] = s - (omega * t)
        rho[i] = (r_hat[0].T @ r[i])

        beta = (rho[i]/rho[i - 1]) * (alpha/omega)
        p[i] = r[i] + (beta * (p[i - 1] - (omega * v)))
    
    return x[-1]

def l2_regularized_cross_entropy_loss(outputs, labels, model, reg_lambda=0.01):
    cross_entropy_loss = F.cross_entropy(outputs, labels)
    l2_reg = sum(torch.norm(param)**2 for param in model.parameters())
    return cross_entropy_loss + reg_lambda * l2_reg

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

def train_model(model, trainloader, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_batches = 0
        for inputs, labels in trainloader:
            outputs = model(inputs)
            loss = l2_regularized_cross_entropy_loss(outputs, labels, model)
            total_loss += loss.item()
            total_accuracy += accuracy(outputs, labels)
            total_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        avg_loss = total_loss / total_batches
        avg_accuracy = total_accuracy / total_batches
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1}: Loss = {avg_loss:.2f}, Accuracy = {avg_accuracy:.2f}%, Time = {elapsed_time:.2f} seconds')

def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Ensure input images are PIL Images
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_path = '/Users/cba/Desktop/cs_6787/final_project/CS6787FinalProject/data/cifar-10-batches-py'
    train_dataset = CIFAR10Dataset(root_dir=dataset_path, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = CIFAR10Net()
    train_model(model, train_loader)

if __name__ == "__main__":
    main()
