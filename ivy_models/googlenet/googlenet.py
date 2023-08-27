# global
import ivy
import ivy_models
from ivy_models.base import BaseSpec, BaseModel
from ivy_models.googlenet.layers import (
    ConvBlock, Inception, Auxiliary,
)

import sys
sys.path.append("/ivy_models/log_sys")
from pf import pf


class GoogLeNetSpec(BaseSpec):
    def __init__(
        self, num_classes=1000, training=False, dropout=0.4, aux_dropout=0.7, data_format="NCHW"
    ):
        if not training:
            dropout=0
            aux_dropout=0
        super(GoogLeNetSpec, self).__init__(
            num_classes=num_classes,
            dropout=dropout,
            aux_dropout=aux_dropout,
            data_format=data_format,
        )


class GoogLeNet(BaseModel):
    """
    Inception-V1 (GoogLeNet) architecture.

    Args::
        num_classes (int): Number of output classes. Defaults to 1000.
        v (ivy.Container): Unused parameter. Can be ignored.

    """

    def __init__(
        self,
        num_classes=1000,
        training=False,
        dropout=0.4,
        aux_dropout=0.7,
        data_format="NCHW",
        spec=None,
        v=None,
    ):
        if not training:
            dropout=0
            aux_dropout=0
        self.spec = (
            spec
            if spec and isinstance(spec, GoogLeNetSpec)
            else GoogLeNetSpec(
                num_classes=num_classes,
                dropout=dropout,
                aux_dropout=aux_dropout,
                data_format=data_format,
            )
        )
        super(GoogLeNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        # N x 3 x 224 x 224
        pf(f'GoogLeNet | build | input shape is: (1, 224, 224, 3)')
        self.conv1 = ConvBlock(3, 64, [7, 7], (2,2), padding="SAME", data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: | done 1/21')
        self.pool1 = ivy.MaxPool2D([3, 3], 2, "SAME", data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 2/21')
        self.conv2 = ConvBlock(64, 64, [1, 1], 1, padding="SAME", data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 3/21')
        self.conv3 = ConvBlock(64, 192, [3, 3], 1, padding="SAME", data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 4/21')
        # self.pool3 = ivy.MaxPool2D([3, 3], 2, "SAME", data_format="NCHW")
        self.pool3 = ivy.MaxPool2D([3, 3], 2, "SAME", data_format=self.spec.data_format)

        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 5/21')
        self.inception3A = Inception(192, 64, 96, 128, 16, 32, 32, data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 6/21')
        self.inception3B = Inception(256, 128, 128, 192, 32, 96, 64, data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 7/21')
        # self.pool4 = ivy.MaxPool2D([3, 3], 2, "SAME", data_format="NCHW")
        self.pool4 = ivy.MaxPool2D([3, 3], 2, "SAME", data_format=self.spec.data_format)

        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 8/21')
        self.inception4A = Inception(480, 192, 96, 208, 16, 48, 64, data_format=self.spec.data_format)

        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 9/21')
        self.aux4A = Auxiliary(512, self.spec.num_classes, self.spec.aux_dropout, data_format=self.spec.data_format)

        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 10/21')
        self.inception4B = Inception(512, 160, 112, 224, 24, 64, 64, data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 11/21')
        self.inception4C = Inception(512, 128, 128, 256, 24, 64, 64, data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 12/21')
        self.inception4D = Inception(512, 112, 144, 288, 32, 64, 64, data_format=self.spec.data_format)

        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 13/21')
        self.aux4D = Auxiliary(528, self.spec.num_classes, self.spec.aux_dropout, data_format=self.spec.data_format)

        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 14/21')
        self.inception4E = Inception(528, 256, 160, 320, 32, 128, 128, data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 15/21')
        # self.pool5 = ivy.MaxPool2D([3, 3], 2, "SAME", data_format="NCHW")
        self.pool5 = ivy.MaxPool2D([3, 3], 2, "SAME")

        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 16/21')
        self.inception5A = Inception(832, 256, 160, 320, 32, 128, 128, data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 17/21')
        self.inception5B = Inception(832, 384, 192, 384, 48, 128, 128, data_format=self.spec.data_format)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 18/21')
        self.pool6 = ivy.AdaptiveAvgPool2d([1, 1])

        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 19/21')
        self.spec.dropout = ivy.Dropout(self.spec.dropout)
        pf(f'GoogLeNet | build | shape is: (1, 224, 224, 3) | done 20/21')
        self.fc = ivy.Linear(1024, self.spec.num_classes)
        pf(f'GoogLeNet | build | output shape is: (1, 224, 224, 3) | done 21/21')

    @classmethod
    def get_spec_class(self):
        return GoogLeNetSpec

    def _forward(self, x, data_format=None):
        # this check is needed becoz we alaways assume that the input image format will
        # be in NHWC format ie, ivy's tensor format, so if we are using NCHW then my model's build part will configure itself 
        # for NCHW format BUT for image we have to permute it as well to NCHW by permuting becoz we assume it to be
        # in NHWC bydefault.
        
        
        # data_format = data_format if data_format else self.spec.data_format
        # if data_format == "NCHW":
        #     x = ivy.permute_dims(x, (0, 2, 3, 1))
        
        pf(f'GoogLeNet | forward | input shape is: | done -1/23')
        out = self.conv1(x)
        pf(f'GoogLeNet | forward | input shape is: | done 0/23')
        out = self.pool1(out)
        pf(f'GoogLeNet | forward | input shape is: | done 1/23')
        out = self.conv2(out)
        pf(f'GoogLeNet | forward | input shape is: | done 2/23')
        out = self.conv3(out)
        pf(f'GoogLeNet | forward | input shape is: | done 3/23')
        out = self.pool3(out)
        pf(f'GoogLeNet | forward | input shape is: | done 4/23')
        out = self.inception3A(out)
        pf(f'GoogLeNet | forward | input shape is: | done 5/23')
        out = self.inception3B(out)
        pf(f'GoogLeNet | forward | input shape is: | done 6/23')
        out = self.pool4(out)
        pf(f'GoogLeNet | forward | input shape is: | done 7/23')
        out = self.inception4A(out)

        pf(f'GoogLeNet | forward | input shape is: | done 8/23')
        aux1 = self.aux4A(out)

        pf(f'GoogLeNet | forward | input shape is: | done 9/23')
        out = self.inception4B(out)
        pf(f'GoogLeNet | forward | input shape is: | done 10/23')
        out = self.inception4C(out)
        pf(f'GoogLeNet | forward | input shape is: | done 11/23')
        out = self.inception4D(out)

        pf(f'GoogLeNet | forward | input shape is: | done 12/23')
        aux2 = self.aux4D(out)

        pf(f'GoogLeNet | forward | input shape is: | done 13/23')
        out = self.inception4E(out)
        pf(f'GoogLeNet | forward | input shape is: | done 14/23')
        out = self.pool5(out)
        pf(f'GoogLeNet | forward | input shape is: | done 15/23')
        out = self.inception5A(out)
        pf(f'GoogLeNet | forward | input shape is: | done 16/23')
        out = self.inception5B(out)

        pf(f'GoogLeNet | forward | input shape is: | done 17/23')
        out = ivy.permute_dims(out, (0, 3, 1, 2))
        pf(f'GoogLeNet | forward | input shape is: | done 18/23')
        out = self.pool6(out)

        pf(f'GoogLeNet | forward | input shape is: | done 19/23')
        out = ivy.flatten(out, start_dim=1)
        pf(f'GoogLeNet | forward | input shape is: | done 20/23')
        out = self.spec.dropout(out)
        pf(f'GoogLeNet | forward | input shape is: | done 22/23')
        out = self.fc(out)
        pf(f'GoogLeNet | forward | input shape is: | done 23/23')

        return out, aux1, aux2

def test_GoogLeNet():
    ivy.set_backend('torch')
    # N x 224 x 224 x 3 
    random_test_tensor = ivy.random_normal(shape=(1, 224, 224, 3))
    pf(f"test_GoogLeNet | random_test_tensor shape is: {random_test_tensor.shape}")

    block = GoogLeNet(data_format="NHWC")
    block(random_test_tensor)
    # N x 1 x 1000
    pf("test_GoogLeNet | Test Successfull!")

test_GoogLeNet()


def _inceptionNet_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv/weight"]
    new_mapping = new_key
    if any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping


def inceptionNet_v1(pretrained=True, training=False, num_classes=1000, dropout=0, data_format="NCHW"):
    """InceptionNet-V1 model"""
    model = GoogLeNet(num_classes=num_classes, training=training, dropout=dropout, data_format=data_format)
    if pretrained:
        url = "https://download.pytorch.org/models/googlenet-1378be20.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_inceptionNet_torch_weights_mapping,
        )
        model.v = w_clean
    return model
