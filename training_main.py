import argparse
import os
from sched import scheduler
import torch
import torch.optim as optim

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset_small/', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--masked_data', type=str, default='bird_dataset_small_masked/', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='how many epochs to wait before saving model')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
# torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from src.data import data_transform_small
from src.dataset import Dataset, get_list_files

train_files, val_files, _ = get_list_files(args.data, args.masked_data)
print("Loading data from {}".format(args.data))

train_dataset = Dataset(train_files, transform=data_transform_small)
val_dataset = Dataset(val_files, transform=data_transform_small)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from src.models import ResNetFeatures, NN

features_model = ResNetFeatures()
features_model.eval()

model = NN(in_layer=2*2048)


if use_cuda:
    print('Using GPU')
    features_model.cuda()
    model.cuda()

else:
    print('Using CPU')
all_params = list(model.parameters()) + list(features_model.parameters())
print("There is {} parameters in the model".format(sum([p.numel() for p in all_params])))
optimizer = optim.Adam(all_params, lr=args.lr) #, momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

def train(epoch):
    model.train()
    for batch_idx, (data, mask, target) in enumerate(train_loader):
        if use_cuda:
            data, mask, target = data.cuda(), mask.cuda(), target.cuda()
        optimizer.zero_grad()
        data = features_model(data)
        mask = features_model(mask)
        # stack the two images
        data = torch.cat((data, mask), 1)
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, mask, target in val_loader:
            if use_cuda:
                data, mask, target = data.cuda(), mask.cuda(), target.cuda()
            optimizer.zero_grad()
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

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
    
    return validation_loss, correct/len(val_loader.dataset)

criteria = 0
for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation_loss, accuracy = validation()

    if accuracy > criteria:
        criteria = accuracy

        print('Saving model best scores at epoch {}'.format(epoch))
        torch.save(model.state_dict(), os.path.join(args.experiment, 'resnet_masked_best_model_221_ter.pth'))


    scheduler.step(validation_loss)

print("The best validation accuracy is {}".format(criteria))
