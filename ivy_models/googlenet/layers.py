import ivy
import sys

sys.path.append("/workspaces/models_Ivy_Sark42/log_sys")
from pf import pf


class ConvBlock(ivy.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super(ConvBlock, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv = ivy.Conv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            with_bias=False,
            data_format="NCHW",
        )
        self.bn = ivy.BatchNorm2D(self.out_channels, eps=0.001, data_format="NCS")

    def _forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = ivy.relu(x)
        return x


class Inception(ivy.Module):
    def __init__(
        self,
        in_channels,
        num1x1,
        num3x3_reduce,
        num3x3,
        num5x5_reduce,
        num5x5,
        pool_proj,
    ):
        self.in_channels = in_channels
        self.num1x1 = num1x1
        self.num3x3_reduce = num3x3_reduce
        self.num3x3 = num3x3
        self.num5x5_reduce = num5x5_reduce
        self.num5x5 = num5x5
        self.pool_proj = pool_proj
        super(Inception, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv_1x1 = ConvBlock(
            self.in_channels, self.num1x1, kernel_size=[1, 1], stride=1, padding=0
        )

        self.conv_3x3 = ConvBlock(
            self.in_channels,
            self.num3x3_reduce,
            kernel_size=[1, 1],
            stride=1,
            padding=0,
        )
        self.conv_3x3_red = ConvBlock(
            self.num3x3_reduce, self.num3x3, kernel_size=[3, 3], stride=1, padding=1
        )

        self.conv_5x5 = ConvBlock(
            self.in_channels,
            self.num5x5_reduce,
            kernel_size=[1, 1],
            stride=1,
            padding=0,
        )
        self.conv_5x5_red = ConvBlock(
            self.num5x5_reduce, self.num5x5, kernel_size=[3, 3], stride=1, padding=1
        )

        self.pool_proj_conv = ConvBlock(
            self.in_channels, self.pool_proj, kernel_size=[1, 1], stride=1, padding=0
        )

    def _forward(self, x):
        # unit testing using ==> """self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)"""
        pf(
            "#######################################################################################"
        )
        # 1x1
        pf(f"the input shape should be (1, 28, 28, 192) | and it is: {ivy.shape(x)}")
        conv_1x1 = self.conv_1x1(x)

        # 3x3
        pf(f"the input shape should be () | and it is: {ivy.shape(x)}")
        conv_3x3 = self.conv_3x3(x)

        pf(f"the input shape should be () | and it is: {ivy.shape(x)}")
        conv_3x3_red = self.conv_3x3_red(conv_3x3)

        # 5x5
        pf(f"the input shape should be () | and it is: {ivy.shape(x)}")
        conv_5x5 = self.conv_5x5(x)

        pf(f"the input shape should be () | and it is: {ivy.shape(x)}")
        conv_5x5_red = self.conv_5x5_red(conv_5x5)

        # pool_proj
        pf(f"the input shape should be () | and it is: {ivy.shape(x)}")
        pool_proj = ivy.max_pool2d(x, [3, 3], 1, 1, ceil_mode=True, data_format="NCHW")

        pf(f"the input shape should be () | and it is: {ivy.shape(x)}")
        pool_proj = self.pool_proj_conv(pool_proj)

        pf(f"the input shape should be () | and it is: {ivy.shape(conv_1x1)}")
        pf(f"the input shape should be () | and it is: {ivy.shape(conv_3x3_red)}")
        pf(f"the input shape should be () | and it is: {ivy.shape(conv_5x5_red)}")
        pf(f"the input shape should be (1, 1024) | and it is: {ivy.shape(pool_proj)}")
        ret = ivy.concat([conv_1x1, conv_3x3_red, conv_5x5_red, pool_proj], axis=1)
        pf(f"the output shape should be (1, 28, 28, 256) | and it is: {ivy.shape(x)}")
        pf(
            "#######################################################################################"
        )

        return ret


class Auxiliary(ivy.Module):
    def __init__(self, in_channels, num_classes):
        self.in_channels = in_channels
        self.num_classes = num_classes
        super(Auxiliary, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv = ConvBlock(self.in_channels, 128, [1, 1], 1, 0)

        self.fc1 = ivy.Linear(2048, 1024)

        self.dropout = ivy.Dropout(0.7)

        self.fc2 = ivy.Linear(1024, self.num_classes)
        self.softmax = ivy.Softmax()

    def _forward(self, x):
        # adap avg pool layer
        # pf(f"the input shape for permute_dim layer should be (1, 14, 14, 512) | and it is: {ivy.shape(x)}")
        # out = ivy.permute_dims(x, (0, 3, 1, 2))

        pf(
            f"the input shape for adapAvgPool layer should be (1, 512, 14, 14) | and it is: {ivy.shape(x)}"
        )
        out = ivy.adaptive_avg_pool2d(x, [4, 4])

        # pf(f"the input shape for permute_dim layer should be (1, 14, 14, 512) | and it is: {ivy.shape(x)}")
        # out = ivy.permute_dims(out, (0, 2, 3, 1))

        # conv
        pf(
            f"the input shape for conv layer should be (1, 4, 4, 512) | and it is: {ivy.shape(out)}"
        )
        out = self.conv(out)

        # flatten
        pf(
            f"the input shape for flatten layer should be (1, 4, 4, 128) | and it is: {ivy.shape(out)}"
        )
        out = ivy.flatten(out, start_dim=1)

        # fc1
        pf(
            f"the input shape for fc1 layer should be (1, 2048) | and it is: {ivy.shape(out)}"
        )
        out = self.fc1(out)

        pf(
            f"the input shape for relu layer should be (1, 1024) | and it is: {ivy.shape(out)}"
        )
        out = ivy.relu(out)

        # dropout
        pf(
            f"the input shape for dropout layer should be (1, 1024) | and it is: {ivy.shape(out)}"
        )
        out = self.dropout(out)

        # fc2
        pf(
            f"the input shape for fc2 layer should be (1, 1024) | and it is: {ivy.shape(out)}"
        )
        out = self.fc2(out)
        pf(
            f"the output shape from fc2 layer should be (1, 1000) | and it is: {ivy.shape(out)}"
        )

        return out
