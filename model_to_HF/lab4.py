def _model(model_repo_name):
    import ivy

    ivy.set_backend("torch")
    from ivy_models.googlenet import inceptionNet_v1
    from ivy_models_tests import helpers

    model = inceptionNet_v1(pretrained=False)
    img = helpers.load_and_preprocess_img(
        "/workspaces/models_Ivy_Sark42/images/dog.jpg",
        256,
        224,
        data_format="NCHW",
        to_ivy=True,
    )
    model = model.load_from_huggingface(
        repo_id=f"unifyai/{model_repo_name}",
        config_path="config.json",
        model_path="googlenet.pkl",
        weights_path="weights.hdf5",
        repo_type="model",
        token=None,
        revision=None,
        safe_serialization=False,
        load_model_object=True,
    )
    logits, _, _ = model(img)
    # print(logits)

    import wget

    wget.download(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Logits of the top 3 classes are: ivy.array([0.73900217, 0.10972812, 0.08693198])
    # Categories of the top 3 classes are: ['clumber', 'beacon', 'digital watch']

    output = ivy.softmax(logits)  # pass the image to the model
    classes = ivy.argsort(output[0], descending=True)[:3]  # get the top 3 classes
    logits = ivy.gather(output[0], classes)  # get the logits
    print("Indices of the top 3 classes are:", classes)
    print("Logits of the top 3 classes are:", logits)
    print(
        "Categories of the top 3 classes are:",
        [categories[i] for i in classes.to_list()],
    )


# def load_weights():


import subprocess

# Define the shell commands as a single string
commands = """
pip install git+https://github.com/unifyai/ivy.git
git clone https://github.com/unifyai/models.git --depth 1
cd models/ && pip install .
pip install wget
"""

# Use subprocess to run the shell commands
process = subprocess.Popen(
    commands,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    executable="/bin/bash",  # Specify the shell (bash in this case)
)

# Wait for the process to complete
process.wait()

# Check the return code to see if any errors occurred
if process.returncode == 0:
    print("Commands executed successfully.")
    _model("googlenet")
else:
    print("Commands failed with error:")
    print(process.stderr.read())
