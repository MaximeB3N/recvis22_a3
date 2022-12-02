from src.box.models import *
from src.box.utils import *

import os
import torch
from torchvision import transforms
from torch.autograd import Variable

from tqdm.notebook import trange
from PIL import Image


def main():
    config_path='config/box/yolov3.cfg'
    weights_path='config/box/yolov3.weights'
    class_path='config/box/coco.names'
    img_size=416
    conf_thres=0.4
    nms_thres=0.3

    # Load model and weights
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    model.cuda()
    model.eval()
    classes = load_classes(class_path)

    data = "data/bird_dataset/"

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
            # print(folder)
            os.makedirs("bird_dataset_small/" + name + folder, exist_ok=True)
            # break


    not_crop = crop_data(model, train_files, classes, img_size=img_size, conf_thres=conf_thres, nms_thres=nms_thres)

    # save the list of files that were not cropped
    with open('bird_dataset_small/not_crop_train.txt', 'w') as f:
        for item in not_crop:
            f.write(item + "\n")

        
    not_crop = crop_data(model, val_files, classes, img_size=img_size, conf_thres=conf_thres, nms_thres=nms_thres)

    # save the list of files that were not cropped
    with open('bird_dataset_small/not_crop_val.txt', 'w') as f:
        for item in not_crop:
            f.write(item + "\n")

    not_crop = crop_data(model, test_files, classes, img_size=img_size, conf_thres=conf_thres, nms_thres=nms_thres)

    # save the list of files that were not cropped
    with open('bird_dataset_small/not_crop_test.txt', 'w') as f:
        for item in not_crop:
            f.write(item + "\n")


def detect_image(model, img, img_size=416, conf_thres=0.4, nms_thres=0.3):
    # scale and pad image
    # check if img is a numpy array
    Tensor = torch.cuda.FloatTensor

    if isinstance(img, np.ndarray):
        w, h = img.shape[:2]
        # convert it to PIL image
        img = Image.fromarray(img)
        # print("Image converted to PIL image")
    else:
        w, h = img.size[0], img.size[1]
        # print("Image is already a PIL image")

    ratio = min(img_size/w, img_size/h)
    imw = round(w * ratio)
    imh = round(h * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def get_crop_image(model, img, classes, img_size=416, conf_thres=0.4, nms_thres=0.3):
        # load image and get detections
    detections = detect_image(model, img, img_size=img_size, conf_thres=conf_thres, nms_thres=nms_thres)
    # print ('Inference Time: %s' % (inference_time))

    # Get bounding-box colors

    img = np.array(img)

    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    if detections is not None:
        i = 0 
        # browse detections and draw bounding boxes
        for x1, y1, x2, y2, _, _, cls_pred in detections:
            if classes[int(cls_pred)] == "bird":
                if i > 1:
                    print("more than one bird")

                i+= 1

                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                img_crop = img[int(y1):int(y1+box_h), int(x1):int(x1+box_w)]
        return img_crop

    else:
        print("no bird")
        return None



def crop_data(model, list_files, classes, img_size=416, conf_thres=0.4, nms_thres=0.3):
    not_crop = []
    for i in trange(len(list_files)):
        file = list_files[i]
       
        img = Image.open(file)

        new_path = file.replace("bird_dataset", "bird_dataset_small")

        if os.path.exists(new_path):
            continue
        
        try:
            img_crop = get_crop_image(model, img, classes, img_size=img_size, conf_thres=conf_thres, nms_thres=nms_thres)
        except:
            not_crop.append(file)
            print("get crop error")
            print(file)
            continue

        if img_crop is not None:
            try:
                img_crop = Image.fromarray(img_crop)
            except:
                print("conversion error")
                print(file)
                not_crop.append(file)
                continue
            
            img_crop.save(new_path)


        else:
            print(file)
            not_crop.append(file)

    return not_crop


if __name__ == '__main__':
    main()