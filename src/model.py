import torch.nn as nn
import torch.nn.functional as F

class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, X):
        x = self.pool1(F.relu(self.conv1(X))) # 1x28x28 --> convolution --> 32x28x28 -->  maxpool --> 32x14x14
        x = self.pool2(F.relu(self.conv2(x))) # 32x14x14 --> convolution --> 64x14x14 --> maxpool --> 64x7x7
        x = x.view(-1, 64 * 7 * 7) # 64x7x7 --> 3136
        x = F.relu(self.fc1(x)) # 3136 --> 128
        x = self.fc2(x) # 128 --> 10
        return x