from torch import nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(SeparableConv2d, self).__init__()
        # This is also equivalent to a group convolution with groups set to 1
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# Residual Depth-Wise Separable Convolutions
class RDWSC(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(RDWSC, self).__init__()

        self.main_path = nn.Sequential(
            SeparableConv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            SeparableConv2d(output_channels, output_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_channels),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
        )

        self.shortcut_path = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, (1, 1), stride=(2, 2)),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        shortcut = self.shortcut_path(x)
        main = self.main_path(x)
        output = shortcut + main
        return output

class MiniXception(nn.Module):
    def __init__(self, num_classes=7):
        super(MiniXception, self).__init__()

        self.base_layers = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), (1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3, 3), stride=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.module1 = RDWSC(input_channels=8, output_channels=16)
        self.module2 = RDWSC(input_channels=16, output_channels=32)
        self.module3 = RDWSC(input_channels=32, output_channels=64)
        self.module4 = RDWSC(input_channels=64, output_channels=128)

        # Final output layer
        self.output_conv = nn.Conv2d(128, num_classes, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.output_conv(x)
        x = x.mean(axis=[-1, -2])  # Global Average Pooling
        return x
