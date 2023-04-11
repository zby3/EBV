import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import argparse

parser.add_argument('--data', type=str,help='input path')
parser.add_argument('--mpath', type=str,help='model path')
args = parser.parse_args()

# define image augmentations
transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0,translate=(5/224,5/224)),
    transforms.ToTensor()
])

# load test set
test_set = datasets.ImageFolder(os.path.join(args.data,'test'), transform = transformations)
test_loader = torch.utils.data.DataLoader(test_set, batch_size =256, shuffle=True,num_workers=8,pin_memory=True)

# load model
PATH=args.mpath
model=models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

# evaluate the model
confusion_matrix = torch.zeros(3, 3)
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
normal_loose, normal_dense, tumor = confusion_matrix.diag()/confusion_matrix.sum(1)

print('Overall Acc on test set: {:.6f}'.format(correct/total))
print('Accuracy in normal_loose: {} \tAccuracy in normal_dense: {} \tAccuracy in tumor: {}'.format(normal_loose, normal_dense,tumor))
print(confusion_matrix)
