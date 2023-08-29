# global
import ivy
import ivy_models
from ivy_models.base import BaseSpec, BaseModel
from ivy_models.googlenet.layers import (
    ConvBlock,
    Inception,
    Auxiliary,
)

import sys

sys.path.append("/ivy_models/log_sys")
from pf import pf


class GoogLeNetSpec(BaseSpec):
    def __init__(
        self,
        num_classes=1000,
        training=False,
        dropout=0.4,
        aux_dropout=0.7,
        data_format="NCHW",
    ):
        if not training:
            dropout = 0
            aux_dropout = 0
        super(GoogLeNetSpec, self).__init__(
            num_classes=num_classes,
            dropout=dropout,
            aux_dropout=aux_dropout,
            data_format=data_format,
        )


class GoogLeNet(ivy.Module):
    def __init__(
        self,
        training=False,
        num_classes=1000,
        dropout=0.4,
        aux_dropout=0.7,
        data_format="NCHW",
        spec=None,
        v=None,
    ):
        self.spec = (
            spec
            if spec and isinstance(spec, GoogLeNetSpec)
            else GoogLeNetSpec(
                training=training,
                num_classes=num_classes,
                dropout=dropout,
                aux_dropout=aux_dropout,
                data_format=data_format,
            )
        )
        super(GoogLeNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.conv1 = ConvBlock(3, 64, [7, 7], 2, padding=3)
        #         self.pool1 = ivy.max_pool2d([3,3], 2, "SAME", ceil_mode=True)

        self.conv2 = ConvBlock(64, 64, [1, 1], 1, padding=0)
        self.conv3 = ConvBlock(64, 192, [3, 3], 1, padding=1)
        #         self.pool3 = ivy.max_pool2d([3,3], 2, "SAME", ceil_mode=True)

        self.inception3A = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3B = Inception(256, 128, 128, 192, 32, 96, 64)
        #         self.pool4 = ivy.max_pool2d([3,3], 2, "SAME", ceil_mode=True)

        self.inception4A = Inception(480, 192, 96, 208, 16, 48, 64)

        self.aux4A = Auxiliary(512, self.spec.num_classes)

        self.inception4B = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4C = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4D = Inception(512, 112, 144, 288, 32, 64, 64)

        self.aux4D = Auxiliary(528, self.spec.num_classes)

        self.inception4E = Inception(528, 256, 160, 320, 32, 128, 128)
        #         self.pool5 = ivy.max_pool2d([3,3], 2, "SAME", ceil_mode=True)

        self.inception5A = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5B = Inception(832, 384, 192, 384, 48, 128, 128)
        self.pool6 = ivy.AdaptiveAvgPool2d([1, 1])

        self.dropout = ivy.Dropout(0.4)
        self.fc = ivy.Linear(1024, self.spec.num_classes, with_bias=False)

    def _forward(self, x, data_format=None):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NHWC":
            x = ivy.permute_dims(x, (0, 3, 1, 2))
        pf(
            f"the input IMAGE shape should be (1, 224, 224, 3) | and it is: {ivy.shape(x)}"
        )
        out = self.conv1(x)

        # maxpool2d_1
        pf(
            f"the input shape for max_pool2d_1 should be (1, 112, 112 64) | and it is: {ivy.shape(out)}"
        )
        out = ivy.max_pool2d(out, [3, 3], 2, 0, ceil_mode=True, data_format="NCHW")

        pf(
            f"the input shape for conv2 should be (1, 56, 56, 64) | and it is: {ivy.shape(x)}"
        )
        out = self.conv2(out)

        pf(
            f"the input shape for conv3 shdould be (1, 56, 56, 64) | and it is: {ivy.shape(out)}"
        )
        out = self.conv3(out)

        # maxpool2d_2
        pf(
            f"the input shape for maxpool2d_2 shdould be (1, 56, 56, 192) | and it is: {ivy.shape(out)}"
        )
        out = ivy.max_pool2d(out, [3, 3], 2, 0, ceil_mode=True, data_format="NCHW")

        pf(f"the input shape shdould be (1, 28, 28, 192) | and it is: {ivy.shape(out)}")
        out = self.inception3A(out)

        pf(f"the input shape shdould be (1, 28, 28, 256) | and it is: {ivy.shape(out)}")
        out = self.inception3B(out)

        # maxpool2d_3
        pf(
            f"the input shape for maxpool2d_3 shdould be (1, 28, 28, 480) | and it is: {ivy.shape(out)}"
        )
        out = ivy.max_pool2d(out, [3, 3], 2, 0, ceil_mode=True, data_format="NCHW")

        pf(
            f"the input shape for inception4a shdould be (1, 14, 14, 480) | and it is: {ivy.shape(out)}"
        )
        out = self.inception4A(out)

        pf(
            f"the input shape for aux4a shdould be (1, 14, 14, 512) | and it is: {ivy.shape(out)}"
        )
        aux1 = self.aux4A(out)
        pf(
            f"the output shape from aux4a shdould be (1, 1000) | and it is: {ivy.shape(out)}"
        )

        pf(
            f"the input shape for inception4b shdould be (1, 14, 14, 512)| and it is: {ivy.shape(out)}"
        )
        out = self.inception4B(out)

        pf(
            f"the input shape for inception4c shdould be (1, 14, 14, 512) | and it is: {ivy.shape(out)}"
        )
        out = self.inception4C(out)

        pf(
            f"the input shape for inception4 shdould be (1, 14, 14, 512) | and it is: {ivy.shape(out)}"
        )
        out = self.inception4D(out)

        pf(
            f"the input shape for aux4d shdould be (1, 14, 14, 528)  | and it is: {ivy.shape(out)}"
        )
        aux2 = self.aux4D(out)
        pf(
            f"the output shape from aux4a shdould be (1, 1000) | and it is: {ivy.shape(out)}"
        )

        pf(
            f"the input shape for inception4e should be (1, 14, 14, 528) | and it is: {ivy.shape(out)}"
        )
        out = self.inception4E(out)

        # maxpool2d_4
        pf(
            f"the input shape for maxpool2d_4 should be (1, 7, 7, 832) | and it is: {ivy.shape(out)}"
        )
        out = ivy.max_pool2d(out, [2, 2], 2, 0, ceil_mode=True, data_format="NCHW")

        pf(
            f"the input shape for inception5a is (1, 7, 7, 832) | and it is: {ivy.shape(out)}"
        )
        out = self.inception5A(out)

        pf(
            f"the input shape for inception5b shdould be (1, 7, 7, 1024) | and it is: {ivy.shape(out)}"
        )
        out = self.inception5B(out)

        pf(
            f"the input shape for AdpAvgPool should be (1, 1, 1, 1024) | and it is: {ivy.shape(out)}"
        )
        #       out = ivy.reshape(out, (1, 1024, 7, 7))
        # out = ivy.permute_dims(out, (0, 3, 1, 2))
        out = self.pool6(out)

        pf(
            f"the input shape for flatten layer should be (1, 1, 1, 1024) | and it is: {ivy.shape(out)}"
        )
        out = ivy.flatten(out, start_dim=1)

        pf(
            f"the input shape for dropout should be (1, 1024) | and it is: {ivy.shape(out)}"
        )
        out = self.dropout(out)

        pf(
            f"the input shape for fc layer should be (1, 1024) | and it is: {ivy.shape(out)}"
        )
        out = self.fc(out)
        pf(f"final fc output shape should be (1, 1000) | and it is: {ivy.shape(out)}")

        return out, aux1, aux2


def _inceptionNet_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv/weight"]
    new_mapping = new_key
    if any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping


def inceptionNet_v1(
    pretrained=True,
    training=False,
    num_classes=1000,
    dropout=0.4,
    aux_dropout=0.7,
    data_format="NCHW",
):
    """InceptionNet-V1 model"""
    model = GoogLeNet(
        training=training,
        num_classes=num_classes,
        dropout=dropout,
        aux_dropout=aux_dropout,
        data_format=data_format,
    )
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
