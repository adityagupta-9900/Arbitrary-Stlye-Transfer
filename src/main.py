import sys
import torch
import torch.nn as nn

from PIL import Image

from StyleTransfer import ContentStyleLoss, StyleTransfer
from train import trainModel
from utilities import save_tensor_image, processTestImage, NameExtract, Parser, getDataset


parser = Parser()
args = parser.parse_args()

if __name__ == '__main__':
    
    if (args.action == "run"):
        contentImage = args.content_image
        styleImage = args.style_image

        contName = NameExtract(contentImage)
        styleName = NameExtract(styleImage)

        contentImage = processTestImage(Image.open(contentImage)).to(device)
        styleImage = processTestImage(Image.open(styleImage)).to(device)
        
        model = StyleTransfer(device)
        model.load_state_dict(torch.load(args.model))

        styledImage = model(contentImage, styleImage, args.alpha)[0]
        
        save_tensor_image(styledImage, f"../outputs/{contName}_{styleName}.jpg", False)
        print("Style Transfer completed! Please view", f"../outputs/{contName}_{styleName}.jpg")

    elif (args.action == "train"):

        lmbda = 5 if not args.lmbda else int(args.lmbda)
        model = StyleTransfer(device)
        loss_fn = ContentStyleLoss(lmbda).to(device)

        contentTrainPath = args.content_image
        styleTrainPath = args.style_image

        model = trainModel(model, loss_fn, *getDataset(contentTrainPath, styleTrainPath, val=args.val, bs=args.bs), device=device)
    
        