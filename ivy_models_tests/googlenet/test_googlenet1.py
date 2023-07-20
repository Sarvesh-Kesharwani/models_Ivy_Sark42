import os
import ivy
import pytest
import numpy as np
from ivy_models.googlenet import inceptionNet_v1
from ivy_models_tests import helpers

def np_softmax(inputs):
    """apply the softmax on the output"""
    return np.exp(inputs) / np.sum(np.exp(inputs))

@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False])
# @pytest.mark.parametrize("load_weights", [False, True])
def test_inception_v1_img_classification(device, f, fw, batch_shape, load_weights):
    """Test Inception-V1 image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # # Load image
    # img = helpers.load_and_preprocess_img(
    #     os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    # )

    # Load and preprocess the image
    from PIL import Image
    image_path = "/models/images/cat.jpg"
    image = Image.open(image_path)
    input_tensor = helpers.load_and_preprocess_img(image)
    img = input_tensor.unsqueeze(0)

    


    # Create model
    model = inceptionNet_v1(pretrained=load_weights)
    
    # Perform inference
    output = model(img)

    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        output = output[0]
        true_indices = ivy.array([282, 281, 285])
        calc_indices = ivy.argsort(output, descending=True)[:3]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.7069, 0.2663, 0.0231])
        calc_logits = np.take(
            np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
        )

        assert np.allclose(true_logits, calc_logits, rtol=0.005)

