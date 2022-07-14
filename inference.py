import glob
import time

import cv2
import torch

# from torchvision import *
import RotationNet


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


def evaluation(input_folder, rotation_model):
    image_paths = glob.glob(input_folder, recursive=True)
    gt_dict = {label: 0 for label in rotation_model.class_name}
    gt_dict['total'] = 0
    pred_dict = {label: 0 for label in rotation_model.class_name}
    pred_dict['total'] = 0

    for image_path in image_paths:
        print(image_path)
        gt_label = image_path.split('/')[-2]
        image = cv2.imread(image_path)
        start = time.time()
        pred, score = rotation_model.predict(image)
        end = time.time()
        print(end - start)
        gt_dict[gt_label] += 1
        gt_dict['total'] += 1
        if pred == gt_label:
            pred_dict[gt_label] += 1
            pred_dict['total'] += 1
    acc_dict = {label: pred_dict[label] / gt_dict[label] for label in gt_dict.keys() if gt_dict[label] != 0}

    return acc_dict, pred_dict, gt_dict


def main():
    use_cuda = torch.cuda.is_available()
    device = str(torch.device('cuda:0' if use_cuda else 'cpu'))
    model_path = "model/rotation_best.pth"
    model = load_model(model_path, device)

    input_folder = "data_rotation/data/test/**/*jpg"
    acc_dict, pred_dict, gt_dict = evaluation(input_folder, model)
    print(acc_dict)
    print(pred_dict)
    print(gt_dict)


if __name__ == "__main__":
    main()
