import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.mask_conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        return x


def alexnet(num_classes, weights=None):
    if weights is None:
        pretrained_alexnet = 'models/pretrained_alexnet.pth'
        if os.path.exists(pretrained_alexnet):
            model = AlexNet(num_classes=num_classes)
            model.load_state_dict(torch.load(pretrained_alexnet))
        else:
            model = AlexNet()
            state_dict = model_zoo.load_url(model_urls['alexnet'])
            new_sd = dict()
            new_sd['conv1.weight'] = state_dict['features.0.weight']
            new_sd['conv1.bias'] = state_dict['features.0.bias']
            new_sd['conv2.weight'] = state_dict['features.3.weight']
            new_sd['conv2.bias'] = state_dict['features.3.bias']
            new_sd['conv3.weight'] = state_dict['features.6.weight']
            new_sd['conv3.bias'] = state_dict['features.6.bias']
            new_sd['conv4.weight'] = state_dict['features.8.weight']
            new_sd['conv4.bias'] = state_dict['features.8.bias']
            new_sd['conv5.weight'] = state_dict['features.10.weight']
            new_sd['conv5.bias'] = state_dict['features.10.bias']
            new_sd['fc6.weight'] = state_dict['classifier.1.weight']
            new_sd['fc6.bias'] = state_dict['classifier.1.bias']
            new_sd['fc7.weight'] = state_dict['classifier.4.weight']
            new_sd['fc7.bias'] = state_dict['classifier.4.bias']
            new_sd['fc8.weight'] = state_dict['classifier.6.weight']
            new_sd['fc8.bias'] = state_dict['classifier.6.bias']
            model.load_state_dict(new_sd)
            model.fc8 = nn.Linear(model.fc8.in_features, num_classes)
    else:
        model = AlexNet(num_classes=num_classes)
        model.load_state_dict(torch.load(weights))
    return model
