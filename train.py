import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch_optimizer
from torch.utils.data import DataLoader
from torchvision import *

import  RotationNet


def load_model(model_path, device="cpu"):
    net_dict = torch.load(model_path)
    print(net_dict.keys())
    transform = net_dict['transform']
    class_name_dict = net_dict['class_name_dict']
    state_dict = net_dict['model']
    model = RotationNet.RotationNet(class_name_dict, transform)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def imshow(inp, title=None):
    inp = inp.cpu()
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(100)


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


def rand_bbox(size, lam):
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, beta=0.8,
                cutmix_prob=0.35):
    since = time.time()
    device = str(next(model.parameters()).device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    r = np.random.rand(1)
                    if beta > 0 and r < cutmix_prob:
                        # generate mixed sample
                        lam = np.random.beta(beta, beta)
                        rand_index = torch.randperm(inputs.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                        # compute output
                        if phase == 'train':
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)

                            # loss = criterion(outputs, labels)

                        else:
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # print(vars(optimizer_ft))
            # print("learning rate", optimizer_ft.defaults['lr'])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_model(net, class_name_dict, transform, output_path):
    temp_dict = {"model": net.state_dict(), "class_name_dict": class_name_dict, "transform": transform}
    torch.save(temp_dict, output_path)
    pass


def main():
    load_checkpoint = False
    output_model_dir = "model"
    batch_size = {"train": 4, "val": 1}
    use_cuda = torch.cuda.is_available()
    device = str(torch.device('cuda:0' if use_cuda else 'cpu'))

    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop((512, 512), scale=(0.08, 1.0)),
            # transforms.Grayscale(),
            transforms.Resize((512, 512)),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485], [0.229])
        ]),
        'val': transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485], [0.229])
        ]),
    }

    data_dir = '/media/minh/HDD1/ubuntu/code/rotate/data_rotation/data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size[x],
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


    if load_checkpoint:
        model_path = "checkpoints/epoch_100_old.pth"
        net = load_model(model_path, device)
    else:
        net = RotationNet.RotationNet(dataloaders['train'].dataset.class_to_idx, data_transforms['val'])
        net.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = RotationNet.LabelSmoothingLoss(classes=net.num_classes, smoothing=0.1)

    # optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = torch_optimizer.Lamb(net.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.2)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft,  T_max=20)
    net = train_model(net, dataloaders, dataset_sizes, criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
                      num_epochs=11)
    net.to('cpu')
    save_model(net, net.class_name_dict, net.transform, output_path=f"{output_model_dir}/rotation_best.pth")


if __name__ == "__main__":
    main()
