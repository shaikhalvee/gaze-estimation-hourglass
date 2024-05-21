from torch import nn

Pool = nn.MaxPool2d


# No need for below
# def batch_norm(x):
#     return nn.BatchNorm2d(x.size()[1])(x)
# batch normalization layer are typically defined within model architectures that have fixed parameters


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, bn=False, rectified_linear_unit=True):
        super(ConvBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2, bias=True)
        self.rectified_linear_unit = None
        self.batch_norm = None
        if rectified_linear_unit:
            self.rectified_linear_unit = nn.ReLU(inplace=True)
        if bn:
            self.batch_norm = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        assert x.size()[1] == self.input_dim, "{} {}".format(x.size()[1], self.input_dim)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.rectified_linear_unit is not None:
            x = self.rectified_linear_unit(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv1 = ConvBlock(input_dim, output_dim // 2, kernel_size=1, rectified_linear_unit=False)
        self.bn2 = nn.BatchNorm2d(output_dim // 2)
        self.conv2 = ConvBlock(output_dim // 2, output_dim // 2, kernel_size=3, rectified_linear_unit=False)
        self.bn3 = nn.BatchNorm2d(output_dim // 2)
        self.conv3 = ConvBlock(output_dim // 2, output_dim, kernel_size=1, rectified_linear_unit=False)
        self.skip_layer = ConvBlock(output_dim, output_dim, kernel_size=1, rectified_linear_unit=False)
        self.need_skip = input_dim != output_dim

        # if input_dim == output_dim:
        #     self.need_skip = False
        # else:
        #     self.need_skip = True

    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class HourglassLayer(nn.Module):
    def __init__(self, num_hourglasses, feature_dim, b):
        return
