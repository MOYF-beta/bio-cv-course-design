import torch
import torch.nn as nn
import torch.nn.init as init

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=2,stride=1,padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=2,stride=2,padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=2,stride=2,padding=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.linear = nn.Sequential(
            nn.Linear(32 * 2 * 2, 32),
            nn.Sigmoid(),
            nn.Dropout(p=0.4),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        #self.softmax = torch.nn.functional.softmax
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # Initialize the weights using nn.init.kaiming_uniform_
                init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                # Initialize BatchNorm layer weights and biases using nn.init.normal_
                init.normal_(module.weight, mean=0, std=0.01)
                init.constant_(module.bias, 0)

    def forward(self, x):
        if(len(x.shape) == 3):
            x = torch.unsqueeze(x,0)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x