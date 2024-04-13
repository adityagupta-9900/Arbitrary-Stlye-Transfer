import torch
import numpy as np
import torch.nn as nn

class ContentStyleLoss(nn.Module):
    def __init__(self, lam=7.5):
        super().__init__()
        self.lam = lam

    def forward (self, content_in, content_out, styles_in, styles_out):
        contentLoss = torch.norm(content_out - content_in)
        styleLoss = np.sum([
            torch.linalg.norm(torch.mean(styles_out[i], (2, 3)) - torch.mean(styles_in[i], (2,3))) + 
            torch.linalg.norm(torch.std(styles_out[i], axis=(2, 3), unbiased=False) - torch.std(styles_in[i], axis=(2, 3),
            unbiased=False)) for i in range(len(styles_in))
    ])

        return contentLoss + self.lam*styleLoss

class StyleTransfer(nn.Module):
    def __init__(self, device="cpu"):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        self.encoder.load_state_dict(torch.load('../models/vgg_weights'))
        self.encoder = nn.Sequential(*list(self.encoder.children())[:31])

        for i in self.encoder.parameters():
            i.requires_grad = False

        self.encoder = self.encoder.to(device)

        self.decoder = nn.Sequential(
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(512, 256, (3, 3)),
                            nn.ReLU(),
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(256, 256, (3, 3)),
                            nn.ReLU(),
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(256, 256, (3, 3)),
                            nn.ReLU(),
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(256, 256, (3, 3)),
                            nn.ReLU(),
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(256, 128, (3, 3)),
                            nn.ReLU(),
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(128, 128, (3, 3)),
                            nn.ReLU(),
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(128, 64, (3, 3)),
                            nn.ReLU(),
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(64, 64, (3, 3)),
                            nn.ReLU(),
                            nn.ReflectionPad2d((1, 1, 1, 1)),
                            nn.Conv2d(64, 3, (3, 3))).to(device)

        self.style_outputs = []
        self.style_layers = [3, 10, 17, 30] # relu1_1, relu2_1, relu3_1, relu4_1
        for i in self.style_layers:
            self.encoder._modules[str(i)].register_forward_hook(self.style_feature_hook)

    def Adain(self):
        cF, sF = self.contentFeatures, self.styleFeatures

        return  (
            torch.std(sF, axis=(2, 3), unbiased=False).reshape(-1, 512, 1, 1) * 
            (cF - torch.mean(cF, (2, 3)).reshape(-1, 512, 1, 1)) / 
            (torch.std(cF, axis=(2, 3), unbiased=False).reshape(-1, 512, 1, 1) + 1e-4)
            ) + \
            torch.mean(sF, (2, 3)).reshape(-1, 512, 1, 1)

    def style_feature_hook(self, module, input, output):
        self.style_outputs.append(output)

    def forward(self, contentImage, styleImage, alpha=1):
        self.contentFeatures = self.encoder(contentImage)
        self.style_outputs = []
        self.styleFeatures = self.encoder(styleImage)
        self.target = self.Adain()
        self.target = (self.target * alpha) + (self.contentFeatures * (1-alpha))
        return self.decoder(self.target)
