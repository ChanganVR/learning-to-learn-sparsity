import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
from models.alexnet import alexnet
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
import sys

logger = logging.getLogger()


def train(model, data_loaders, dataset_sizes, reg_lambda, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    logger.info('Initial sparsity of conv2: {:.4f}'.format(model.compute_sparsity()))
    for epoch in range(num_epochs):
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterating over data once is one epoch
            for data in data_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # loss = crossentropy + l1 regularization on mask
                loss = criterion(outputs, labels) + model.get_mask().norm(1) * reg_lambda

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        logger.info('Sparsity of conv2: {:.4f}'.format(model.compute_sparsity()))

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


def load_model(network, num_classes, mask_network, binarization_func, frozen_layers):
    if network == 'alexnet':
        model = alexnet(num_classes, mask_network, binarization_func, frozen_layers)
        model.cuda()
    else:
        raise NotImplementedError

    logger.info('Trainable parameters: {}'.format([name for name, p in model.named_parameters() if p.requires_grad]))

    return model


def load_dataset(dataset):
    if dataset == 'dtd':
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        data_dir = 'datasets/dtd_splits'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val', 'test']}
        data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=400,
                                                       shuffle=True, num_workers=4)
                        for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    else:
        raise NotImplementedError

    return data_loaders, dataset_sizes


def main():
    network = 'alexnet'
    dataset = 'dtd'
    num_classes = 47
    reg_lambda = 1e-5
    num_epochs = 100
    learning_rate = 0.001
    step_size = 500
    # 1x1, 3x3, 5x5, no_mask
    mask_network = '1x1'
    # sign, sigmoid
    binarization_func = 'sign'
    frozen_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

    # logging config
    if not os.path.exists('results'):
        os.mkdir('results')
    log_file = 'results/{}_{}_{}_{}_{}_{}.log'.format(network, dataset, reg_lambda, num_epochs, mask_network,
                                                      binarization_func)
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stdout_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    model = load_model(network, num_classes, mask_network, binarization_func, frozen_layers)
    data_loaders, dataset_sizes = load_dataset(dataset)

    criterion = nn.CrossEntropyLoss().cuda()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 100 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)

    best_model, best_acc = train(model, data_loaders, dataset_sizes, reg_lambda,
                                 criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    torch.save(best_model.state_dict(), 'models/masked_alexnet_{:.4f}.pth'.format(best_acc))


if __name__ == '__main__':
    main()
