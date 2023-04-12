import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import numpy as np
from ..utils.helper import val, pat_auc
import argparse

parser.add_argument('--data', type=str,help='input path')
parser.add_argument('--mpath', type=str,help='model path')
args = parser.parse_args()

# define augmentations
transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(5 / 224, 5 / 224)),
    transforms.ToTensor()
])
# load model
PATH=args.mpath
model=models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(PATH))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss()

# load test_set
test_set = datasets.ImageFolder(os.path.join(args.data,'test'), transform=transformations)
accuracy, _ = val(model, test_set, criterion, device)
print('Accuracy on test set (tile-level): {}'.format(accuracy))

# patient-level accuracy
auc=pat_auc(model, test_set, device)
print('Patient-level AUC on test set: {}'.format(auc))
