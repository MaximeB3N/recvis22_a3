import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])


data_transform_no_norm = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor()
])

data_to_tensor = transforms.Compose([
    transforms.ToTensor()
])

data_transform_small = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4493, 0.4575, 0.3962],
                                    std=[0.2601, 0.2581, 0.2701])
])

