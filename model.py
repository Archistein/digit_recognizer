import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from typing import Callable


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False) # 8x26x26
        self.bn1 = nn.BatchNorm2d(8) # 8x26x26
        self.max_pool1 = nn.MaxPool2d(kernel_size=2) # 8x13x13
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3) # 16x11x11
        self.bn2 = nn.BatchNorm2d(16) # 16x11x11
        self.max_pool2 = nn.MaxPool2d(kernel_size=2) # 16x5x5
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3) # 24x3x3

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=24 * 3 * 3, out_features=64, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(self.bn3(x))

        x = self.fc2(x)

        return x


def fit(model: nn.Module, epoch: int, device: torch.device, data_loader: data.DataLoader, 
        optimizer: torch.optim, loss_func: Callable, get_accuracy: Callable):
    
    model.train()

    for i in range(epoch):

        running_loss = 0.0
        for j, batch in enumerate(data_loader):
            X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = loss_func(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if j % 500 == 0:
                with torch.no_grad():
                    print(f'Epoch {i+1}, Epoch Loss: {loss}, Test accuracy: {get_accuracy(model)}')


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = MNIST('data', train=True, transform=ToTensor(), download=True)
    test_data = MNIST('data', train=False, transform=ToTensor(), download=True)

    print(f'Train dataset size: {len(train_data)}')
    print(f'Test dataset size: {len(test_data)}')

    train_data_loader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_data_loader = data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    model = CNN()

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    model.to(device)

    cuda_data = test_data_loader.dataset.data.view(-1, 1, 28, 28).to(device) / 255
    cuda_targets = test_data_loader.dataset.targets.to(device)

    def get_test_accuracy(model):
        model.eval()
        with torch.no_grad():
            y_pred = model(cuda_data)
            bools = y_pred.argmax(1) == cuda_targets
        model.train()
        return torch.mean(bools.float())

    epoch = 10
    fit(model, epoch, device, train_data_loader, optimizer, loss_func, get_test_accuracy)

    print(f'Test accuracy: {get_test_accuracy(model)}')

    # Save model params
    # torch.save(model.state_dict(), 'parameters.pt')

if __name__ == '__main__':
    main()