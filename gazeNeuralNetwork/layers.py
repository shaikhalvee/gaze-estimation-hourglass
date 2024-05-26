from torch import nn

Pool = nn.MaxPool2d


# No need for below
def batch_norm(x):
    return nn.BatchNorm2d(x.size()[1])(x)
# batch normalization layer are typically defined within model architectures that have fixed parameters


class ConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1,
                 batch_normalization=False, rectified_linear_unit=True):
        super(ConvolutionalLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2, bias=True)
        self.rectified_linear_unit = None
        self.batch_normalization = None
        if rectified_linear_unit:
            self.rectified_linear_unit = nn.ReLU(inplace=True)
        if batch_normalization:
            self.batch_normalization = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        assert x.size()[1] == self.input_dim, "{} {}".format(x.size()[1], self.input_dim)
        x = self.conv(x)
        if self.batch_normalization is not None:
            x = self.batch_normalization(x)
        if self.rectified_linear_unit is not None:
            x = self.rectified_linear_unit(x)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv1 = ConvolutionalLayer(input_dim, output_dim // 2, kernel_size=1, rectified_linear_unit=False)
        self.bn2 = nn.BatchNorm2d(output_dim // 2)
        self.conv2 = ConvolutionalLayer(output_dim // 2, output_dim // 2, kernel_size=3, rectified_linear_unit=False)
        self.bn3 = nn.BatchNorm2d(output_dim // 2)
        self.conv3 = ConvolutionalLayer(output_dim // 2, output_dim, kernel_size=1, rectified_linear_unit=False)
        self.skip_layer = ConvolutionalLayer(output_dim, output_dim, kernel_size=1, rectified_linear_unit=False)

        # if input_dim == output_dim:
        #     self.need_skip = False
        # else:
        #     self.need_skip = True
        self.need_skip = input_dim != output_dim

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
    def __init__(self,
                 num_of_downsampling_steps,
                 num_of_input_output_features,
                 batch_normalization=None,
                 increase_feature=0):
        super(HourglassLayer, self).__init__()
        num_of_features = num_of_input_output_features + increase_feature
        # Upper branch
        self.upper_branch_1 = ResidualLayer(num_of_input_output_features, num_of_input_output_features)
        # Lower branch
        self.pooling_layer = Pool(2, 2)
        self.lower_branch_1 = ResidualLayer(num_of_input_output_features, num_of_features)
        self.num_downsample_steps = num_of_downsampling_steps
        # Recursive Hourglass
        if self.num_downsample_steps > 1:
            self.lower_branch_2 = HourglassLayer(num_of_input_output_features - 1,
                                                 num_of_features,
                                                 batch_normalization=batch_normalization)
        else:
            self.lower_branch_2 = ResidualLayer(num_of_features, num_of_features)
        self.lower_branch_3 = ResidualLayer(num_of_input_output_features, num_of_features)

    def forward(self, x):
        upper_branch_1 = self.upper_branch_1(x)
        pooling_layer = self.pooling_layer(x)
        lower_branch_1 = self.lower_branch_1(pooling_layer)
        lower_branch_2 = self.lower_branch_2(lower_branch_1)
        lower_branch_3 = self.lower_branch_3(lower_branch_2)
        upper_branch_2 = nn.functional.interpolate(lower_branch_3, x.shape[2:], mode='bilinear')
        return upper_branch_1 + upper_branch_2
