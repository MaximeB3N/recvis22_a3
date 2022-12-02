import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
from torchvision import datasets
import torch
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
import matplotlib.pyplot as plt
import numpy as np

from model import Net, ResNet, NN, NN2, ResNetFeatures
from dataset import Dataset, get_list_files

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset_small/', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--masked_data', type=str, default='bird_dataset_small_masked/', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='submission/kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--val', type=int, default=0,
                    help="evaluate on validation set instead of test set")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

def main():
    state_dict = torch.load(args.model)
    features_model = ResNetFeatures()
    model = NN2(in_layer=4096)
    model.load_state_dict(state_dict)
    
    if use_cuda:
        print('Using GPU')
        features_model.cuda()
        model.cuda()
    else:
        print('Using CPU')

    model.eval()
    features_model.eval()
    from data import data_transform_small_eval


    if args.val != 0:
        batch_size = 64
        _, val_files, _ = get_list_files(args.data, args.masked_data)
        val_dataset = Dataset(val_files, transform=data_transform_small_eval)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=os.cpu_count())

        validation_loss = validation(val_loader, model, features_model, use_cuda)

        print('Validation loss: {}'.format(validation_loss))



    else: 
        test_dir = args.data + '/test_images/mistery_category'

        def pil_loader(path):
            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(path, 'rb') as f:
                with Image.open(f) as img:
                    return img.convert('RGB')

        _, val_files, test_files = get_list_files(args.data, args.masked_data)
        test_dataset = Dataset(test_files, transform=data_transform_small_eval)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                    shuffle=False, num_workers=os.cpu_count())

        val_dataset = Dataset(val_files, transform=data_transform_small_eval)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                    shuffle=False, num_workers=os.cpu_count())

        output_file = open(args.outfile, "w")
        output_file.write("Id,Category\n")
        for data, mask, name in tqdm(test_loader):
            if use_cuda:
                data, mask = data.cuda(), mask.cuda()
            
            # batch size is 1
            # data = data.unsqueeze(0)
            # mask = mask.unsqueeze(0)

            data = features_model(data)
            mask = features_model(mask)
            # stack the two images
            data = torch.cat((data, mask), 1)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            # print(output.data.max(1, keepdim=True))
            # print(pred)
            # break
            # output_file.write("%s,%d\n" % (name[0], pred))
            output_file.write("%s,%d\n" % (name[0][:-4], pred))

        output_file.close()

        print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
            



def validation(val_loader, model, features_model, use_cuda):
    model.eval()
    validation_loss = 0
    correct = 0
    # list to store the predictions to plot the confusion matrix
    predictions = []
    # list to store the ground truth to plot the confusion matrix
    ground_truth = []
    with torch.no_grad():
        for data, mask, target in val_loader:
            if use_cuda:
                data, mask, target = data.cuda(), mask.cuda(), target.cuda()

            data = features_model(data)
            mask = features_model(mask)
            # stack the two images
            data = torch.cat((data, mask), 1)
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            predictions.extend(pred.cpu().numpy().tolist())
            ground_truth.extend(target.cpu().numpy().tolist())
        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
    
    # plot the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    heatmap(cm, annot=True, fmt="d")
    plt.title("Training confusion matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    return validation_loss


if __name__ == '__main__':
    main()