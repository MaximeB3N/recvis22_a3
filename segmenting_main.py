import os
import torch
import PIL.Image as Image
from mmseg.apis import inference_segmentor, init_segmentor
import numpy as np
from tqdm import trange

def main():
    config = "config/segmentation/pspnet_r101-d8_480x480_40k_pascal_context.py"
    checkpoint = "config/segmentation/pspnet_r101-d8_480x480_40k_pascal_context_20200911_211210-bf0f5d7c.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = "data/bird_dataset_small/"

    model = init_segmentor(config, checkpoint, device=device)

    data = dataset
    new_dataset = "data/bird_dataset_small_masked/"
    os.makedirs(new_dataset, exist_ok=True)  

    train_folders = [f for f in os.listdir(data + 'train_images') if f != '.DS_Store']
    train_files = []
    for folder in train_folders:
        if folder != ".DS_Store":
            names = os.listdir(data + 'train_images/' + folder)
            for name in names:
                train_files.append(data + 'train_images/' + folder + '/' + name)

    val_folders = [ f for f in os.listdir(data + 'val_images') if f != ".DS_Store"]
    val_files = []
    for folder in val_folders:
        if folder != ".DS_Store":
            names = os.listdir(data + 'val_images/' + folder)
            for name in names:
                val_files.append(data + 'val_images/' + folder + '/' + name)


    test_folders = [ f for f in os.listdir(data + 'test_images/') if f != ".DS_Store"]
    test_files = []
    for folder in test_folders:
        if folder != ".DS_Store":
            names = os.listdir(data + 'test_images/' + folder)
            for name in names:
                test_files.append(data + 'test_images/' + folder + '/' + name)

    for folders, name in zip([train_folders, test_folders, val_folders], ["train_images/", "test_images/", "val_images/"]):
        for folder in folders:
            os.makedirs(new_dataset + name + folder, exist_ok=True)

    not_crop = masked_backgrounds(model, train_files)

    # save the list of files that were not cropped
    with open(new_dataset + 'not_crop_train.txt', 'w') as f:
        for item in not_crop:
            f.write(item + "\n")

    not_crop = masked_backgrounds(model, val_files)

    # save the list of files that were not cropped
    with open(new_dataset + 'not_crop_val.txt', 'w') as f:
        for item in not_crop:
            f.write(item + "\n")

    not_crop = masked_backgrounds(model, test_files)

    # save the list of files that were not cropped
    with open(new_dataset + 'not_crop_test.txt', 'w') as f:
        for item in not_crop:
            f.write(item + "\n")


def get_masked_image(model, img, class_id=7):
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    result = inference_segmentor(model, img)
    mask = result[0]
    mask = mask != class_id
    img_crop = img.copy()
    img_crop[mask] = 0
    return img_crop

def masked_background(model, file):
    img = np.array(Image.open(file))

    new_path = file.replace("bird_dataset_small", "bird_dataset_small_masked")

    if os.path.exists(new_path):
        return ""
    
    try:
        img_crop = get_masked_image(model, img)
    except:
        print("get crop error")
        print(file)
        return file

    if img_crop is not None:
        try:
            img_crop = Image.fromarray(img_crop)
        except:
            print("conversion error")
            print(file)
            return file
        
        img_crop.save(new_path)
        return ""

    else:
        print(file)
        return file

def masked_backgrounds(model, list_files):
    not_crop = []

    for i in trange(len(list_files)):
        file = list_files[i]
        res = masked_background(model, file)
        if res != "":
            not_crop.append(res)        

    return not_crop


if __name__ == "__main__":
    main()