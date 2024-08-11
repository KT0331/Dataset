import os
import json

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageNet
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def get_class_label():
    print('Get Training Image Start')
    dataset = ImageFolder(root='./imagenet/train/')
    print('Get Training Image Success')

    train_class2idx  = dataset.class_to_idx
    train_image_list = list(map(lambda x: os.path.relpath(x[0], './imagenet/train/'), dataset.imgs))
    train_label_list = list(map(lambda x: x[1], dataset.imgs))

    annotations = {
                    "filenames": train_image_list,
                    "labels": train_label_list
                  }
    
    json_path = os.path.join('./imagenet/train/', "annotations.json")

    with open(json_path, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)

    print('Get Validation Image Start')
    dataset = ImageFolder(root='./imagenet/val/')
    print('Get Validation Image Success')

    val_class2idx  = dataset.class_to_idx
    val_image_list = list(map(lambda x: os.path.relpath(x[0], './imagenet/val/'), dataset.imgs))
    val_label_list = list(map(lambda x: x[1], dataset.imgs))

    annotations = {
                    "filenames": val_image_list,
                    "labels": val_label_list
                  }
    
    json_path = os.path.join('./imagenet/val/', "annotations.json")

    with open(json_path, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)

    if train_class2idx == val_class2idx:
        print('Output Image Label Success')
    else:
        print('Train Class2idx not Same as Validation Class2idx')

if __name__ == "__main__":
    get_class_label()
