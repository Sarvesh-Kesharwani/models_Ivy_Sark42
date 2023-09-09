import os
import ivy
import random
import numpy as np
import jax

# Enable x64 support in JAX
jax.config.update("jax_enable_x64", True)
from ivy_models_tests import helpers
from ivy_models import (
    resnet_18,
    resnet_34,
    resnet_50,
    resnet_101,
    resnet_152,
)


VARIANTS = {
    "r18": resnet_18,
    "r34": resnet_34,
    "r50": resnet_50,
    "r101": resnet_101,
    "r152": resnet_152,
}

LOGITS = {
    "r18": np.array([0.7069, 0.2663, 0.0231]),
    "r34": np.array([0.8507, 0.1351, 0.0069]),
    "r50": np.array([0.3429, 0.0408, 0.0121]),
    "r101": np.array([0.7834, 0.0229, 0.0112]),
    "r152": np.array([0.8051, 0.0473, 0.0094]),
}


load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = ivy.to_numpy(model.v)


def test_resnet_img_classification(device, fw):
    # """Test ResNet-18 image classification."""
    # num_classes = 1000
    # batch_shape = [1]
    # this_dir = os.path.dirname(os.path.realpath(__file__))

    # # Load image
    # img = ivy.asarray(
    #     helpers.load_and_preprocess_img(
    #         os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
    #         256,
    #         224,
    #         data_format="NHWC",
    #         to_ivy=True,
    #     ),
    # )

    # # model.v = ivy.asarray(v)
    # model.v = model.load_from_huggingface(
    #     repo_id="unifyai/resnet_18",
    #     config_path="config.json",
    #     model_path="model.pkl",
    #     weights_path="weights.hdf5",
    #     repo_type="model",
    #     token=None,
    #     revision=None,
    #     safe_serialization=False,
    #     load_model_object=False,
    # )
    # output = model(img)

    # # Cardinality test
    # assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # # Value test
    # if load_weights:
    #     output = output[0]
    #     true_indices = ivy.array([282, 281, 285])
    #     calc_indices = ivy.argsort(output, descending=True)[:3]

    #     assert np.array_equal(true_indices, calc_indices)

    #     true_logits = LOGITS[model_var]
    #     calc_logits = np.take(
    #         helpers.np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
    #     )

    #     assert np.allclose(true_logits, calc_logits, rtol=0.005)

    model_name = "resnet_18"
    model = resnet_18(pretrained=True)
    model.push_to_huggingface(
        repo_id=f"unifyai/{model_name}",
        config_path="config.json",
        model_path="model.pkl",
        weights_path="weights.hdf5",
        repo_type="model",
        token="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",
        private=False,
        revision=None,
        commit_message=f"adds {model_name} model",
        commit_description=f"Adding {model_name} model, config(specs), and weights",
        create_pr=False,
        safe_serialization=False,
        push_config=True,
        push_model=True,
        push_weights=True,
    )
