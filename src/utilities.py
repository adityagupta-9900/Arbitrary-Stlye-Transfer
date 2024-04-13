# Utility functions
import cv2
import torch
import argparse
import numpy as np

from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder

def read_image(path, size=None, gray=False):
    img = cv2.imread(path)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if size != None:
        img = cv2.resize(img, size)
    return img

scale = lambda x : (255 * (x - x.min())) / (x.max() - x.min())

def save_tensor_image(img, name, de_normalize=True):

    if de_normalize:
        img = torch.round(torch.dstack((scale(img[0]), scale(img[1]), scale(img[2])))).type(torch.uint8)
        img = Image.fromarray(img.cpu().detach().numpy())
        img.save(name)
    else:
        save_image(img, name)

processTestImage = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.unsqueeze(0))
])

processTrainImage = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

NameExtract = lambda path: ExtensionRemove(path[path.rfind('/')+1:])
ExtensionRemove = lambda name: name[:name.rfind('.')]

def Parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="Action to perform [run, train, run_mult, run_alpha]")
    return parser

def getDataset(contentPath, stylePath, val=0.2, bs=64):

    contentTrain = ImageFolder(contentPath, transform=processTrainImage)
    styleTrain = ImageFolder(stylePath, transform=processTrainImage)

    sz = min(len(contentTrain), len(styleTrain))

    contentTrain, contentVal = getLoaders(contentTrain, size=sz, val=val, bs=bs)
    styleTrain, styleVal = getLoaders(styleTrain, size=sz, val=val, bs=bs)
    return contentTrain, contentVal, styleTrain, styleVal

def getLoaders(dataset, size, val=0.2, bs=64):
    
    indices = list(range(size))
    split = int(np.floor(val * size))
    np.random.seed(1)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=bs, sampler=valid_sampler)

    return train_loader, val_loader