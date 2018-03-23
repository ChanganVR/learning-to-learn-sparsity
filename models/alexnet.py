import os
import torch
import logging
import glob
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


logger = logging.getLogger()
__all__ = ['AlexNet', 'alexnet']
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, ):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
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

        # x = self.conv2(x)
        mask = F.sigmoid(self.mask_cnn(self.conv2.weight))
        if not self.training:
            # discrete binarization for val or test
            # threshold the importance of single weight, not the weight itself
            mask = (mask >= 0.5).float()
        masked_weights = torch.mul(mask, self.conv2.weight)
        x = F.conv2d(x, masked_weights, self.conv2.bias, padding=2)

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

    def compute_sparsity(self):
        mask = F.sigmoid(self.mask_cnn(self.conv2.weight))
        masked_weights = torch.mul(mask, self.conv2.weight)
        non_zeros = torch.sum(mask.data.ge(0.5))
        size = torch.prod(torch.FloatTensor([x for x in masked_weights.size()]))
        return 1 - non_zeros / size

    def get_mask(self):
        mask = F.sigmoid(self.mask_cnn(self.conv2.weight))
        return mask


def alexnet(num_classes, weights=None, finetuning=False):
    if finetuning:
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
        model.fc6 = nn.Linear(model.fc6.in_features, model.fc6.out_features)
        model.fc7 = nn.Linear(model.fc7.in_features, model.fc7.out_features)
        model.fc8 = nn.Linear(model.fc8.in_features, num_classes)
        for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            for param in model.__getattr__(layer).parameters():
                param.requires_grad = False
    else:
        if weights is None:
            weights = 'models/finetuned_alexnet_0.5543.pth'
        model = AlexNet(num_classes=num_classes)
        model.load_state_dict(torch.load(weights))
        model.mask_cnn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1)
        )
        for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
            for param in model.__getattr__(layer).parameters():
                param.requires_grad = False
    return model
