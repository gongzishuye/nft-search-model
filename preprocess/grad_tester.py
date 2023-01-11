import glob
import os

import torch
from PIL import Image


# x = torch.tensor([1], requires_grad=False)
# is_train = True

# with torch.set_grad_enabled(is_train):
#     y = x * 2

# print(y.requires_grad)
# print(x.requires_grad)

base_dir = '/home/ec2-user/workspace/data/train/*'
for d in glob.glob(base_dir):
    print(d)
    for f in os.listdir(d):
        file = os.path.join(d, f)
        img = Image.open(file).convert('RGB')
        h, w = img.size
        if h != w: 
            print(img.size, file)

