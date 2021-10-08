import torch
import torch.nn as nn


class Model1(nn.Module):
    """
    Model1 Shallow Network with 3 Convolutions followed by
    fully connected layers
    """

    def __init__(self):

        super(Model1, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class ResBlock(nn.Module):

    """
    Residual Block with 2 Convolutions and Batchnorms
    followed by a skip-connection
    """

    def __init__(self, channels):

        super(ResBlock, self).__init__()

        self.relu = nn.ReLU()

        self.channels = channels

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):

        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x+res


class Model2(nn.Module):
    """
    Deeper Network Compared to model1
    More Convolutions with skip connections
    """

    def __init__(self):

        super(Model2, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.res1 = ResBlock(64)

        self.res2 = ResBlock(128)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.res3 = ResBlock(256)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.res4 = ResBlock(512)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(12800, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.res2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.res3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.res4(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.fc5(x)

        return x


class Model3(nn.Module):
    """
    Similar Network to Model2 but has an LSTM unit
    """

    def __init__(self):

        super(Model3, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.res1 = ResBlock(64)

        self.res2 = ResBlock(128)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.res3 = ResBlock(256)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1536, 1024)

        self.rnn = nn.LSTM(1024, 512, dropout=0.4)

        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x, hidden):

        x = self.conv1(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.res3(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = x.unsqueeze(1)
        x, hidden = self.rnn(x, hidden)
        x = self.fc3(x)
        x = self.relu(x)

        x = x.squeeze(1)

        x = self.fc4(x)

        return x, hidden

    def init_hidden(self, device):

        h0 = torch.zeros(1, 1, 512).to(device)
        c0 = torch.zeros(1, 1, 512).to(device)

        return (h0, c0)
