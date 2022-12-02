import numpy as np
import torch
import os
import PIL.Image as Image

class Dataset:
    def __init__(self, list_files, transform=None):
        self.list_files = list_files
        self.transform = transform

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):

        img = Image.open(self.list_files[idx][0])

        img_masked = Image.open(self.list_files[idx][1])

        class_id = self.list_files[idx][2]

        if not self.transform is None:
            img = self.transform(img)
            img_masked = self.transform(img_masked)
        return img, img_masked, class_id


def get_list_files(root_path, root_path_masked):
    train_folders = [f for f in os.listdir(root_path + 'train_images') if f != '.DS_Store']
    train_folders = sorted(train_folders)
    classes = {train_folders[i]: i for i in range(len(train_folders))}
    
    train_files = []
    for folder in train_folders:
        if folder != ".DS_Store":
            names = os.listdir(root_path + 'train_images/' + folder)
            for name in names:
                train_files.append((root_path + 'train_images/' + folder + '/' + name, 
                        root_path_masked + 'train_images/' + folder + '/' + name,
                        classes[folder]))

    val_folders = [ f for f in os.listdir(root_path + 'val_images') if f != ".DS_Store"]
    val_files = []
    for folder in val_folders:
        if folder != ".DS_Store":
            names = os.listdir(root_path + 'val_images/' + folder)
            for name in names:
                val_files.append((root_path + 'val_images/' + folder + '/' + name, 
                        root_path_masked + 'val_images/' + folder + '/' + name,
                        classes[folder]))


    test_folders = [ f for f in os.listdir(root_path + 'test_images/') if f != ".DS_Store"]
    test_files = []
    for folder in test_folders:
        if folder != ".DS_Store":
            names = os.listdir(root_path + 'test_images/' + folder)
            for name in names:
                test_files.append((root_path + 'test_images/' + folder + '/' + name, 
                        root_path_masked + 'test_images/' + folder + '/' + name,
                        name))

    
    return train_files, val_files, test_files

    
        