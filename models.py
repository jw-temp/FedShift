import torch.nn as nn
import torch.nn.init as init


class CNN2(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN2, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(5, 5),
            padding=1,
            stride=1,
            bias=False,
        )
        # init.xavier_normal_(self.conv1.weight)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=(5, 5),
            padding=1,
            stride=1,
            bias=False,
        )
        # init.xavier_normal_(self.conv2.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1, return_indices=True)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=(hidden_channels * 2) * (8 * 8),
            out_features=num_hiddens,
            bias=False,
        )
        # init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(
            in_features=num_hiddens, out_features=num_classes, bias=False
        )
        # init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x, _ = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x, _ = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


class CNN3(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN3, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(5, 5),
            padding=1,
            stride=1,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=(5, 5),
            padding=1,
            stride=1,
            bias=False,
        )

        # Third convolutional layer
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels * 4,  # Adjust the number of output channels
            kernel_size=(3, 3),  # You can adjust kernel size, padding, and stride
            padding=1,
            stride=1,
            bias=False,
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=1, return_indices=True)

        self.flatten = nn.Flatten()

        # Adjust the input dimensions of fc1
        self.fc1 = nn.Linear(
            in_features=(hidden_channels * 4) * (5 * 5),
            out_features=num_hiddens,
            bias=False,
        )

        self.fc2 = nn.Linear(
            in_features=num_hiddens, out_features=num_classes, bias=False
        )

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x, _ = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x, _ = self.maxpool2(x)

        x = self.activation(self.conv3(x))  # Apply conv3
        x, _ = self.maxpool3(x)  # Apply maxpool3
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


class FNN(nn.Module):
    def __init__(self, name, input_dim, num_hiddens, num_classes):
        super(FNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        # CIFAR10 images are 32x32x3 = 3072 flattened
        self.input_dim = input_dim
        self.fc1 = nn.Linear(
            in_features=self.input_dim, out_features=num_hiddens, bias=False
        )
        # init.xavier_normal_(self.fc1.weight)  # Uncomment for xavier initialization

        self.fc2 = nn.Linear(
            in_features=num_hiddens, out_features=num_hiddens, bias=False
        )
        # init.xavier_normal_(self.fc2.weight)  # Uncomment for xavier initialization

        self.fc3 = nn.Linear(
            in_features=num_hiddens, out_features=num_classes, bias=False
        )
        # init.xavier_normal_(self.fc3.weight)  # Uncomment for xavier initialization

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten the input data
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNBN(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNNBN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(5, 5),
            padding=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)  # BatchNorm after conv1

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=(5, 5),
            padding=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)  # BatchNorm after conv2

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=(hidden_channels * 2) * (8 * 8),
            out_features=num_hiddens,
            bias=False,
        )
        self.bn_fc1 = nn.BatchNorm1d(num_hiddens)  # BatchNorm before FC1

        self.fc2 = nn.Linear(
            in_features=num_hiddens, out_features=num_classes, bias=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Apply BatchNorm after conv1
        x = self.activation(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)  # Apply BatchNorm after conv2
        x = self.activation(x)
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn_fc1(x)  # Apply BatchNorm before FC1
        x = self.activation(x)
        x = self.fc2(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
