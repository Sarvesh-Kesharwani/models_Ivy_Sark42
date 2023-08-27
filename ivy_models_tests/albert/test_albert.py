import os
import pytest
import numpy as np
from ivy_models import albert_base_v1


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_albert(device, fw, batch_shape, load_weights):
    """Test Albert Base Sequence Classification"""

    this_dir = os.path.dirname(os.path.realpath(__file__))

    img_path = os.path.join(this_dir, "albert_inputs.npy")
    inputs = np.load(img_path, allow_pickle=True).tolist()
    model = albert_base_v1(load_weights)

    # inputs = {k: ivy.asarray(v) for k, v in inputs.items()}
    # logits = model(**inputs)["pooler_output"]
    # assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_dims])

    # if load_weights:
    #     logits_path = os.path.join(this_dir, "albert_pooled_output.npy")
    #     ref_logits = np.load(logits_path)
    #     assert np.allclose(ref_logits, ivy.to_numpy(logits), rtol=0.005, atol=0.0005)
