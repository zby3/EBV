import torch
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd
from tqdm import tqdm

def train(model,train_set,optimizer,criterion,device):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    model.train()
    train_loss=0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's running loss
        train_loss += loss.item()*labels.size(0)
        train_loss /= len(train_set)
    return model, train_loss

def val(model,val_set,criterion,device):
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    val_loss = 0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model(inputs)
            # Calculate Loss
            loss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_set)
    accuracy = correct/total
    return accuracy, val_loss

def pat_auc(model, test_set, device):
    """
    calculate patient-level AUC on test set
    """
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    pred = []  # tile-level prediction
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            pred.extend(predicted.tolist())

    df = pd.DataFrame(test_loader.dataset.imgs, columns=['tile', 'target'])  # summarize the result table
    df['pat'] = df['tile'].apply(lambda x: x.split('/')[-1][0:12])
    df['pred'] = pred

    # calculate patient-level score (proportion of POS tiles)
    pat=df.pat.unique()
    pat_pred, pat_true=[],[]  #  patient-level scores, true label
    for i in tqdm(len(pat)):
        tmp=df[df['pat']==pat[i]]
        pat_pred.append(tmp['pred'].sum()/len(tmp))
        pat_true.append(tmp['target'].iloc[0])

    # calculate patient-level AUC
    fpr, tpr, thresholds = roc_curve(pat_true, pat_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc


