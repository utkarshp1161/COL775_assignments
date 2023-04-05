import torch
import torch.nn as nn

'''
This implementation assumes that the input images are RGB images of size 32x32. The first convolutional layer takes in 3 input channels (corresponding to the R, G, and B channels), and outputs 16 feature maps. The _make_layer function creates a residual block with two convolutional layers and a skip connection, which is applied n times for each of the three sets of hidden layers.

The output layer consists of an adaptive average pooling layer that averages the feature maps over the spatial dimensions, followed by a fully connected layer with r units, where r is the number of classes.

You can create an instance of the ResNet model by calling the resnet_6n_2 function with the desired values of n and r. For example:
'''


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups= 32, eps=1e-5, affine=True):
        super(MyGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups

        assert C % G == 0, "Number of channels must be divisible by number of groups"

        # Reshape x to be (N, G, C // G, H, W)
        x = x.view(N, G, C // G, H, W)

        # Compute mean and variance over channels and spatial dimensions
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True)

        # Normalize x using mean and variance
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Reshape x back to its original shape
        x = x.view(N, C, H, W)

        # Apply weight and bias if affine is True
        if self.affine:
            x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)

        return x









class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MyGroupNorm(num_channels = out_channels, num_groups= 8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MyGroupNorm(num_channels = out_channels, num_groups= 8)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, n, r):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = MyGroupNorm(num_channels = 16, num_groups= 8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, r)
        

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        out = self.conv(x)
        #print("debug--batch--norm")
        #print("debug--batch--norm", out.shape)
        out = self.bn(out)
        #print("debug--batch--norm", out.shape)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
