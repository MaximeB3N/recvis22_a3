import os


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
