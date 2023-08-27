import ivy
import sys

sys.path.append("/ivy_models/log_sys")
from pf import pf



class ConvBlock(ivy.Module):
    """
    Conv block used in the GoogLeNet architecture.

    Args::
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size ([list | tuple]): kernel_shape for the block.
        stride (Optional[ivy.Module]): Stride value for the block.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, data_format="NCHW"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
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
        self.bn = ivy.BatchNorm2D(self.out_channels, eps=0.001, data_format=self.data_format)
        self.relu = ivy.ReLU()

    def _forward(self, x):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NCHW":
            x = ivy.permute_dims(x, (0, 2, 3, 1))
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def test_ConvBlock():
    ivy.set_backend('torch')
    # conv1
    # N x 224 x 224 x 3
    random_test_tensor = ivy.random_normal(shape=(1, 224, 224, 3))
    pf(f"test_ConvBlock | random_test_tensor shape is: {random_test_tensor.shape}")

    block = ConvBlock(3, 64, [7,7], (2,2), 3)
    block(random_test_tensor)
    # N x 112 x 112 x 64
    pf("test_ConvBlock | Test Successfull!")
    pf("||")

test_ConvBlock()


class Inception(ivy.Module):
    """
    Inception block used in the GoogLeNet architecture.

    Args::
        in_channels (int): Number of input channels.
        num1x1 (int): Number of num1x1 channels.
        num3x3_reduce (int): Number of num3x3_reduce channels.
        num3x3 (int): Number of num3x3 channels.
        num5x5_reduce (int): Number of num5x5_reduce channels.
        num5x5 (int): Number of num5x5 channels.
        pool_proj (int): Number of pool_proj channels.
    """

    def __init__(
        self,
        in_channels,
        num1x1,
        num3x3_reduce,
        num3x3,
        num5x5_reduce,
        num5x5,
        pool_proj,
        data_format="NCHW",
    ):
        self.in_channels = in_channels
        self.num1x1 = num1x1
        self.num3x3_reduce = num3x3_reduce
        self.num3x3 = num3x3
        self.num5x5_reduce = num5x5_reduce
        self.num5x5 = num5x5
        self.pool_proj = pool_proj,
        self.data_format = data_format,
        super(Inception, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv_1x1 = ivy.Sequential(
            ConvBlock(
                self.in_channels,
                self.num1x1,
                kernel_size=[1, 1],
                stride=1,
                padding="SAME",
                data_format = self.data_format,
            )
        )

        self.conv_3x3 = ivy.Sequential(
            ConvBlock(
                self.in_channels,
                self.num3x3_reduce,
                kernel_size=[1, 1],
                stride=1,
                padding="SAME",
                data_format = self.data_format,
            ),
            ConvBlock(
                self.num3x3_reduce,
                self.num3x3,
                kernel_size=[3, 3],
                stride=1,
                padding="SAME",
                data_format = self.data_format,
            ),
        )

        self.conv_5x5 = ivy.Sequential(
            ConvBlock(
                self.in_channels,
                self.num5x5_reduce,
                kernel_size=[1, 1],
                stride=1,
                padding="SAME",
                data_format = self.data_format,
            ),
            ConvBlock(
                self.num5x5_reduce,
                self.num5x5,
                kernel_size=[3, 3],
                stride=1,
                padding="SAME",
                data_format = self.data_format,
            ),
        )

        self.pool_proj = ivy.Sequential(
            ivy.MaxPool2D([3, 3], 1, "SAME", data_format = self.data_format,),
            ConvBlock(
                self.in_channels,
                self.pool_proj,
                kernel_size=[1, 1],
                stride=1,
                padding="SAME",
                data_format = self.data_format,
            ),
        )

    def _forward(self, x):
        conv_1x1 = self.conv_1x1(x)

        conv_3x3 = self.conv_3x3(x)

        conv_5x5 = self.conv_5x5(x)

        pool_proj = self.pool_proj(x)

        return ivy.concat([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)


def test_Inception():
    ivy.set_backend('torch')
    # inception3a
    # N x 192 x 28 x 28
    random_test_tensor = ivy.random_normal(shape=(1, 28, 28, 192))
    pf(f"test_Inception | random_test_tensor shape is: {random_test_tensor.shape}")

    block = Inception(192, 64, 96, 128, 16, 32, 32)
    block(random_test_tensor)
    # N x 256 x 28 x 28
    pf("test_Inception | Test Successfull!")
    pf("||")

test_Inception()


class Auxiliary(ivy.Module):
    """
    Auxiliary block used in the GoogLeNet architecture.

    Args::
        in_channels (int): Number of input channels.
        num_classes (int): Number of channels in last fc layer.
    """

    def __init__(self, in_channels, num_classes, aux_dropout=0.7, data_format="NCHW"):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.aux_dropout = aux_dropout
        self.data_format = data_format
        super(Auxiliary, self).__init__()

    def _build(self, *args, **kwargs):
        self.pool = ivy.AvgPool2D([4, 4], 3, "VALID", data_format=self.data_format)
        self.conv = ConvBlock(self.in_channels, 128, [1, 1], 1, "SAME", data_format=self.data_format)

        self.fc1 = ivy.Linear(2048, 1024)
        self.relu = ivy.ReLU()

        self.dropout = ivy.Dropout(self.aux_dropout)

        self.fc2 = ivy.Linear(1024, self.num_classes)
        self.softmax = ivy.Softmax()

    def _forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        out = ivy.flatten(out, start_dim=1)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out

def test_Auxiliary():
    ivy.set_backend('torch')
    # N x 512 x 14 x 14
    random_test_tensor = ivy.random_normal(shape=(1, 14, 14, 512))
    pf(f"test_Auxiliary | random_test_tensor shape is: {random_test_tensor.shape}")

    block = Auxiliary(512, 1000, 0)
    block(random_test_tensor)
    # N x 1 x 1000
    pf("test_Auxiliary | Test Successfull!")
    pf("||")

test_Auxiliary()