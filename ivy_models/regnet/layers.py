from typing import Optional
from collections import OrderedDict
import ivy


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


# TODO: add support for more activation functions
#  when we have theme implemented in ivy.
ACT2CLS = {
    "gelu": ivy.GELU(),
    "mish": ivy.Mish(),
    "relu": ivy.ReLU(),
    "relu6": ivy.ReLU6(),
    "sigmoid": ivy.Sigmoid(),
    "silu": ivy.SiLU(),
    "tanh": ivy.Tanh(),
}
ACT2FN = ClassInstantier(ACT2CLS)


class RegNetConvLayer(ivy.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
    ):
        super().__init__()
        self.convolution = ivy.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.normalization = ivy.BatchNorm2D(out_channels)
        self.activation = (
            ACT2FN[activation] if activation is not None else ivy.Identity()
        )

        ivy.GEGLU()

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state

def test_RegNetConvLayer3():
    from pprint import pprint
    random_test_tensor = ivy.random_normal(shape=(1, 256, 256, 3))
    pprint(f"random_test_tensor shape is: {random_test_tensor.shape}")
    
    block = RegNetConvLayer(
            self.spec.num_channels, config.embedding_size, kernel_size=3, stride=2, activation=config.hidden_act
        )
    block(random_test_tensor)
    display("Test Successfull!")
    
test_RegNetConvLayer3()


class RegNetConvLayer(ivy.Module):
(RegNetSpec,
                    RegNetSpec,
                    RegNetEmbeddings,
                    RegNetShortCut,
                    RegNetSELayer,
                    RegNetYLayer,
                    RegNetStage,
                    RegNetEncoder,
                    RegNet,
                    RegNetForImageClassification
                    )
