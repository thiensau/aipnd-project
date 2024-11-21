import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

def save_checkpoint(model, optimizer, class_to_idx, filepath):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, filepath)
    
def save_checkpoint_resnet(model, optimizer, class_to_idx, filepath):
    # Retrieve input features from the first Linear layer in the Sequential
    if isinstance(model.fc, nn.Sequential):
        # Extract features from the last Linear layer in the Sequential block
        for layer in reversed(model.fc):
            if isinstance(layer, nn.Linear):  # Ensure it's a Linear layer
                output_features = layer.out_features
                input_features = layer.in_features
                break
    else:  # If it's a single Linear layer
        input_features = model.fc.in_features
        output_features = model.fc.out_features

    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'fc': {'input_features': input_features, 'output_features': output_features},
        'architecture': 'resnet50',
        'epochs': args.epochs,
        'hidden_units': args.hidden_units,
    }
    torch.save(checkpoint, filepath)
    
def save_checkpoint_densenet(model, optimizer, class_to_idx, filepath):
    """
    Save a checkpoint for a DenseNet model.

    Args:
        model (torch.nn.Module): The trained DenseNet model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        class_to_idx (dict): A mapping of class labels to indices.
        filepath (str): Path where the checkpoint will be saved.
    """
    checkpoint = {
        'state_dict': model.state_dict(),  # Save model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
        'class_to_idx': class_to_idx,  # Save class-to-index mapping
        'classifier': model.classifier,  # Save the custom classifier
        'architecture': 'densenet121',  # Save model architecture
        'input_size': model.classifier[0].in_features,  # Save classifier input size
        'output_size': model.classifier[-2].out_features,  # Save classifier output size
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='resnet', help='Choose model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=256, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs ')
parser.add_argument('--save_dir', type=str, default='.', help='Save checkpoints')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help = "Use GPU")

args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
print(device)

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

if args.arch == 'densenet':
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, args.hidden_units),  # Add a hidden layer with 256 units
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),   # Output layer for num_classes
        nn.LogSoftmax(dim=1))
elif args.arch == 'resnet':
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1))

print(torch.version.cuda)
print(torch.__version__)

criterion = nn.NLLLoss()

if args.arch == 'densenet':
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
elif args.arch == 'resnet':
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
model.to(device)

epochs = args.epochs
print_every = 40
steps = 0
for e in range(epochs):
    running_loss = 0
    for inputs, labels in train_dataloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                validation_loss = 0
                accuracy = 0
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model.forward(inputs)
                    batch_loss = criterion(outputs, labels)               
                    validation_loss += batch_loss.item()
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(valid_dataloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_dataloader):.3f}")
                running_loss = 0
                model.train()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy= %d %%' % (100 * correct / total))

class_to_idx = train_dataset.class_to_idx
if args.arch == 'densenet':
    save_checkpoint_densenet(model, optimizer, class_to_idx, 'checkpoint_densenet.pth')
elif args.arch == 'resnet':
    save_checkpoint_resnet(model, optimizer, class_to_idx, 'checkpoint_resnet.pth')
    

# python train.py --learning_rate 0.0001 --hidden_units 256 --epochs 5 --arch resnet --gpu
# python train.py --learning_rate 0.0001 --hidden_units 128 --epochs 5 --arch densenet --gpu
