

#!/bin/bash

# Navigate to your Git repository directory
cd /ivy_models/

# Update the repository to the latest changes
git pull

# add prod files
git add /ivy_models/ivy_models_tests/inceptionnetv3/test_inceptionnetv3.py /ivy_models/ivy_models/inceptionnetv3/inceptionnetv3.py /ivy_models/ivy_models/inceptionnetv3/layers.py

# Create a new branch
# git checkout 

# Make some changes to your code
# ...

echo "Git commands executed successfully!"
