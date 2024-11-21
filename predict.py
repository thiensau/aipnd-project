import argparse
import torch
import json
from torch import nn, optim
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))   
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1)) 
    return torch.tensor(np_image, dtype=torch.float32)

def predict(image, model, topk=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    image = image.to(device)

    with torch.no_grad ():
        output = model.forward(image)
        
    output_prob = torch.exp(output)
    
    probs, indeces = output_prob.topk(topk)
    probs = probs.to('cpu').numpy().tolist()[0]
    indeces = indeces.to('cpu').numpy().tolist()[0]
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indeces]
    
    return probs, classes

def load_checkpoint_resnet(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    
    # Initialize the ResNet model
    model = models.resnet50(pretrained=True)
    
    # Freeze feature extractor layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild the classifier (fc) layer
    hidden_units = checkpoint.get('hidden_units', 512)  # Default to 512 if not saved
    num_classes = len(checkpoint['class_to_idx'])      # Number of output classes
    model.fc = nn.Sequential(
        nn.Linear(2048, hidden_units),  # ResNet-50's fc layer has 2048 input features
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1)
    )
    
    # Load state dict and class-to-index mapping
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def load_checkpoint_densenet(filepath):
    """
    Load a checkpoint and rebuild the DenseNet model.

    Args:
        filepath (str): Path to the saved checkpoint.

    Returns:
        model (torch.nn.Module): The reconstructed DenseNet model.
        class_to_idx (dict): The class-to-index mapping.
    """
    checkpoint = torch.load(filepath)
    
    # Load the architecture
    if checkpoint['architecture'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Architecture {checkpoint['architecture']} is not supported.")
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Rebuild the classifier
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])  # Load model weights
    
    # Get class-to-index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to image')
parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
parser.add_argument('--arch', type=str, default='resnet', help='Choose model architecture')
parser.add_argument('--top_k', type=int, default=1, help='top K most likely classes')
parser.add_argument('--category_names', type=str, default = 'cat_to_name.json', help='Path to JSON file mapping categories')
parser.add_argument('--gpu', action='store_true', help='Use GPU')
args = parser.parse_args()

# load check point
if args.arch == 'densenet':
    model = load_checkpoint_densenet(args.checkpoint)
elif args.arch == 'resnet':
    model = load_checkpoint_resnet(args.checkpoint)

device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
model.to(device)

# Process image 
img = process_image(args.input)
img = img.unsqueeze(0).to(device)

# Predict top K
probs, classes = predict(img, model, args.top_k)
print(probs)
print(classes)

# Map
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[str(cls)] for cls in classes]

# Print result here
for prob, cls in zip(probs, classes):
    print(f'{cls}: {prob*100:.2f}%')
    
plt.figure(figsize = (6,10))
ax = plt.subplot(2,1,1)
image = process_image(args.input)
ax = imshow(image, ax=ax, title=classes[0])
ax.axis('off')

plt.subplot(2,1,2)
sb.barplot(x=probs, y=classes, color=sb.color_palette()[0]);
plt.show()
    

# python predict.py flowers/test/47/image_04993.jpg checkpoint_resnet.pth --category_names cat_to_name.json --top_k 3 --gpu
# python predict.py flowers/test/5/image_05159.jpg checkpoint_resnet.pth --category_names cat_to_name.json --top_k 3 --gpu
# python predict.py flowers/test/37/image_03783.jpg checkpoint_resnet.pth --category_names cat_to_name.json --top_k 5 --gpu
# python predict.py flowers/test/33/image_06454.jpg checkpoint_densenet.pth --arch densenet --category_names cat_to_name.json --top_k 5 --gpu
# python predict.py flowers/test/2/image_05100.jpg checkpoint_densenet.pth --arch densenet --category_names cat_to_name.json --top_k 5 --gpu
# python predict.py flowers/test/51/image_03984.jpg checkpoint_densenet.pth --arch densenet --category_names cat_to_name.json --top_k 5 --gpu