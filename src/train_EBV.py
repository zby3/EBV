import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import splitfolders
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from ..utils.helper import train, val
import argparse

parser.add_argument('--data', type=str,help='input path')
parser.add_argument('--lr', type=float,help='learning rate')
parser.add_argument('--l2', type=float,help='l2 regularization')
parser.add_argument('--non_improve', type=float,help='max epochs for non-decreasing val_loss')
parser.add_argument('--mpath', type=str,help='model path')
parser.add_argument('--epoch', type=int,help='epochs')
args = parser.parse_args()

# define augmentations
transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(5 / 224, 5 / 224)),
    transforms.ToTensor()
])
# load train and val set
train_set = datasets.ImageFolder(os.path.join(args.data,'train'), transform=transformations)
val_set = datasets.ImageFolder(os.path.join(args.data,'val'), transform=transformations)
# load model with fixed weights for first 6 layers
model = models.resnet18(pretrained=True)
child_counter = 0
for child in model.children():
    if child_counter < 7:
        print("child ", child_counter, " was frozen")
        for param in child.parameters():
            param.requires_grad = False
    elif child_counter == 7:
        children_of_child_counter = 0
        for children_of_child in child.children():
            if children_of_child_counter < 1:
                for param in children_of_child.parameters():
                    param.requires_grad = False
                print('child ', children_of_child_counter, 'of child', child_counter, ' was frozen')
            else:
                print('child ', children_of_child_counter, 'of child', child_counter, ' was not frozen')
            children_of_child_counter += 1

    else:
        print("child ", child_counter, " was not frozen")
    child_counter += 1
model.fc = torch.nn.Linear(512, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# define hyperparameters
epochs = args.epoch
non_improve = args.non_improve  # max epochs for non-decreasing val_loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
PATH = args.mpath  # save models to
# train
if __name__ == "__main__":
    count = 0  # Epoch counter for non-decreasing val_loss
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model, train_loss = train(model, train_set, optimizer, criterion, device)
        accuracy, val_loss = val(model, val_set, criterion, device)
        print(
            'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f}'.format(epoch, train_loss,
                                                                                                    val_loss,
                                                                                                    accuracy))
        if val_loss < best_val_loss:
            count = 0
            torch.save(model.state_dict(), PATH)
        else:
            count += 1
        if count >= non_improve:
            break
