import torch.nn as nn

class SNRWDNN(nn.Module):
    def __init__(self, channels, num_of_layers=8):
        super(SNRWDNN, self).__init__()
        features = 32
        layers = []
        kernel_size = 3
        padding = 1
        groups = 32
        
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(features))
        
        for _ in range(num_of_layers):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=False))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        self.snrwdnn = nn.Sequential(*layers)
        
    def forward(self, ipt):
        out = ipt - self.snrwdnn(ipt)
        return out
        
    