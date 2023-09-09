# install ivy_models pkg
! pip install -q ivy
! git clone https://github.com/unifyai/models.git --depth 1

# Installing models package from cloned repository! 😄
! cd models/ && pip install .
! cd ..

! python3 -m pip install torchvision




# 
def upload_to_hf(models_list):
    
    from .. import ivy_models/{}
    from ivy_models.googlenet import inceptionNet_v1
    
    
    model = inceptionNet_v1(pretrained=True)
    model.save_pretrained()

upload_to_hf([])