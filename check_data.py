import os
import numpy as np
import PIL.Image as Image

path1 = "bird_dataset/test_images/mistery_category"
path2 = "bird_dataset_small/test_images/mistery_category"

set1 = set(os.listdir(path1))
set2 = set(os.listdir(path2))

print(len(set1))
print(len(set2))

print(len(set1.intersection(set2)))
print(len(set2.difference(set1)))
print(set2.difference(set1))

# for f in set2.difference(set1):
#     os.remove(path2 + "/" + f)

path = "/users/eleves-b/2019/maxime.bonnin/RecVis/recvis22_a3/bird_dataset_small_masked/train_images/009.Brewer_Blackbird/Brewer_Blackbird_0028_2682.jpg"

img = Image.open(path)
img = np.array(img)
print(img.shape)
img = np.repeat(img[:,:, np.newaxis], 3, axis=2)
print(img.shape)