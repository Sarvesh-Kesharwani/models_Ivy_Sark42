# global
from typing import List, Optional, Type, Union
import builtins

import ivy
import ivy_models
from ivy_models.regnet.layers import (RegNetSpec,
RegNetEmbeddings,
RegNetShortCut,
RegNetSELayer,
RegNetYLayer,
RegNetStage,
RegNetEncoder,
RegNet,
)
from ivy_models.base import BaseSpec, BaseModel
from dataclasses import dataclass


class RegNetSpec(BaseSpec):
    """ResNetSpec class"""

    def __init__(
        self,
        num_channels=3,
        num_labels=None,  #TODO
        embedding_size=32,
        hidden_sizes=[128, 192, 512, 1088],
        depths=[2, 6, 12, 2],
        groups_width=64,
        layer_type="y",
        hidden_act="relu",
        **kwargs,
    ) -> None:
        super(RegNetSpec, self).__init__(
            num_channels=num_channels,
            num_labels=num_labels,
            embedding_size=embedding_size,
            hidden_sizes=hidden_sizes,
            depths=depths,
            groups_width=groups_width,
            layer_type=layer_type,
            hidden_act=hidden_act,
            **kwargs,
        )


class RegNetModel(BaseModel):
    """
    Self-Regulated Network for Image Classification (RegNet) architecture.

    Args::
        embedder: TODO
        encoder: TODO
        pooler: TODO
        v (ivy.Container): Unused parameter. Can be ignored.

    """
    def __init__(
        self,
        embedder: Type[RegNetEmbeddings],
        encoder: Type[RegNetEncoder],
        pooler,
        spec=None,
        v: ivy.Container = None,
    ) -> None:
        self.spec = (
            spec
            if spec and isinstance(spec, RegNetSpec)
            else RegNetSpec(
                embedder, encoder, pooler
            )
        )
        super(RegNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.embedder = RegNetEmbeddings(self.spec)
        self.encoder = RegNetEncoder(self.spec)
        self.pooler = ivy.AdaptiveAvgPool2d((1, 1))

    def _forward(self, pixel_values, output_hidden_states = None, return_dict = None):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.spec.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.spec.use_return_dict

        embedding_output = self.embedder(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return [
            last_hidden_state,
            pooled_output,
            encoder_outputs.hidden_states,
        ]


class RegNetForImageClassification():
    def __init__(
            self,
            num_channels=3,
            embedding_size=32,
            hidden_sizes=[128, 192, 512, 1088],
            depths=[2, 6, 12, 2],
            groups_width=64,
            layer_type="y",
            hidden_act="relu",
            num_labels=None, #TODO
            spec=None,
            v: ivy.Container = None,
    ):
        self.spec = (
            spec
            if spec and isinstance(spec, RegNetSpec)
            else RegNetSpec(num_channels,
                            num_labels,
                            embedding_size,
                            hidden_sizes,
                            depths,
                            groups_width,
                            layer_type,
                            hidden_act,
                            )
                    )
        super(RegNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.num_labels = self.spec.num_labels
        self.regnet = RegNetModel(self.spec)
        self.classifier = ivy.Sequential(
            ivy.Flatten(),
            ivy.Linear(self.spec.hidden_sizes[-1],
                        self.spec.num_labels) if self.spec.num_labels > 0 else ivy.Identity(),
        )

    def _forward(
        self,
        pixel_values=None,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.spec.use_return_dict

        outputs = self.regnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == ivy.int64 or labels.dtype == ivy.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                # TODO: need implementation of ivy.MSELoss()
                loss_fct = ivy.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = ivy.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # TODO: need implementation of ivy.BCEWithLogitsLoss()
                loss_fct = ivy.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return [loss, logits, outputs.hidden_states]
    

def RegNet_for_classification(pretrained=True):
    """RegNet for image classification"""
    model = RegNet()
    if pretrained:
        base_model_prefix = "regnet"
        main_input_name = "pixel_values"
        supports_gradient_checkpointing = True
        
        # this code basically initialized/ loads weights using some logic instead
        # of downloading and loading theme.
        # def _init_weights(self, module):
        #     if isinstance(module, nn.Conv2d):
        #         nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(module.weight, 1)
        #         nn.init.constant_(module.bias, 0)

        # def _set_gradient_checkpointing(self, module, value=False):
        #     if isinstance(module, RegNetModel):
        #         module.gradient_checkpointing = value
    return model
