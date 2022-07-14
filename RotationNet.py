import torch
import torch.nn as nn
from torch.nn import functional
from torchvision import models
from PIL import Image


class RotationNet(nn.Module):

    def __init__(self, class_name_dict, transform):
        super(RotationNet, self).__init__()
        self.class_name_dict = class_name_dict
        self.id2name = {idx: cls_name for cls_name, idx in self.class_name_dict.items()}
        self.class_name = self.class_name_dict.keys()
        self.num_classes = len(self.class_name)
        self.transform = transform
        res_net = models.resnet50(pretrained=True)
        # self.input_layer = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features = nn.Sequential(*list(res_net.children())[:-1])
        # self.av
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        # forward input through convs
        # x = self.input_layer(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, image):
        image_pil = Image.fromarray(image)
        image_tensor = self.transform(image_pil)
        image_tensor = image_tensor.unsqueeze(0)

        if str(next(self.parameters()).device) == "cuda:0":
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            output = self(image_tensor)
            scores = functional.softmax(output, dim=1)
        prob, pred = torch.max(scores, dim=1)
        prob, pred = prob.cpu().numpy(), pred.cpu().numpy()
        return self.id2name[pred[0]], prob[0]

    def batch_predict(self, image, batch_size):
        image_pil = Image.fromarray(image)

        image_tensor = self.transform(image_pil)
        # image_tensor = image_tensor.unsqueeze(0)
        batch = [image_tensor]*batch_size
        image_tensor = torch.stack(batch)

        if str(next(self.parameters()).device) == "cuda:0":
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            output = self(image_tensor)
            scores = functional.softmax(output, dim=1)
            # print(scores)

        prob, pred = torch.max(scores, dim=1)
        prob, pred = prob.cpu().numpy(), pred.cpu().numpy()
        pred = [self.id2name[x] for x in pred]
        return pred, prob


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
