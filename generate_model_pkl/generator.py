from ivy_models.googlenet import inceptionNet_v1


def test_saving():
    model = inceptionNet_v1(pretrained=True)
    model.save_pretrained()
