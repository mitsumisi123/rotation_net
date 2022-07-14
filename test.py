import argparse

import torch
import RotationNet
import cv2

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--image_path', help='directory contain input image')
    parser.add_argument(
        '--batch_size', type=int, help='directory where painted images will be saved')

    args = parser.parse_args()
    return args

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

def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = str(torch.device('cuda:0' if use_cuda else 'cpu'))
    # model_path = "model/rotation_best.pth"
    model_path = args.checkpoint
    image_path = args.image_path
    batch_size = args.batch_size

    model = load_model(model_path, device)
    image = cv2.imread(image_path)
    pred, score = model.batch_predict(image, batch_size)
    print(batch_size)
    print(pred)
    print(score)

if __name__ == '__main__':
    main()