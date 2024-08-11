import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageNet
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def ppm_to_jpg_and_create_annotations(ppm_dir, train_dir, val_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    data_train_list = []
    label_train_list = []
    data_val_list = []
    label_val_list = []
    annotations_train = []
    annotations_val = []

    print(f'Get Training Image Start...')
    dataset = ImageFolder(root=ppm_dir)
    print(f'Get Training Image Success...')

    img_class = dataset.classes

    for mkclass_dir in img_class:
        path = os.path.join(train_dir ,mkclass_dir)
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(val_dir ,mkclass_dir)
        if not os.path.exists(path):
            os.makedirs(path)

    train_image_list = list(map(lambda x: x[0], dataset.imgs))
    train_label_list = list(map(lambda x: x[1], dataset.imgs))

    print(f'Split Dataset Start...')
    train_image_list, val_image_list, train_label_list, val_label_list = \
        train_test_split(train_image_list, train_label_list, test_size=0.2,
                         stratify=train_label_list, random_state=42)
    print(f'Split Dataset Success...')

    # train_set = {'file': train_image_list,
    #              'label': train_label_list}

    # val_set = {'file': val_image_list,
    #            'label': val_label_list}
    
    print(f'Conversion Train Image Start...')
    for (file, label) in zip(train_image_list, train_label_list):
        ppm_path = file
        relpath = os.path.relpath(ppm_path, ppm_dir)
        jpg_filename = relpath.replace('.ppm', '.jpg')
        jpg_path = os.path.join(train_dir, jpg_filename)
        
        # Convert PPM to JPG
        with Image.open(ppm_path) as img:
            img.save(jpg_path, 'JPEG')
        
        # Append to annotations
        data_train_list.append(jpg_filename)
        label_train_list.append(label)

    annotations_train = {
                    "filenames": data_train_list,
                    "labels": label_train_list
                  }
    print(f'Conversion Train Image Success...')
    print(f'Conversion Val Image Start...')
    for (file, label) in zip(val_image_list, val_label_list):
        ppm_path = file
        relpath = os.path.relpath(ppm_path, ppm_dir)
        jpg_filename = relpath.replace('.ppm', '.jpg')
        jpg_path = os.path.join(val_dir, jpg_filename)
        
        # Convert PPM to JPG
        with Image.open(ppm_path) as img:
            img.save(jpg_path, 'JPEG')
        
        # Append to annotations
        data_val_list.append(jpg_filename)
        label_val_list.append(label)

    annotations_val = {
                    "filenames": data_val_list,
                    "labels": label_val_list
                  }
    print(f'Conversion Val Image Success...')

    # Write annotations to a JSON file
    with open(os.path.join(train_dir, 'annotations.json'), 'w') as json_file:
        json.dump(annotations_train, json_file, indent=4)
    print(f'Train Annotation Finished...')

    with open(os.path.join(val_dir, 'annotations.json'), 'w') as json_file:
        json.dump(annotations_val, json_file, indent=4)
    print(f'Val Annotation Finished...')
    
    print(f"Conversion complete...")

# Define your directories here
# ppm_directory = './Final_Training/Images/'
# jpg_train_directory = './train'
# jpg_val_directory = './val'
ppm_directory = '/home/mediarti3/dataset/GTSRB/Final_Training/Images/'
jpg_train_directory = '/home/mediarti3/dataset/GTSRB/train'
jpg_val_directory = '/home/mediarti3/dataset/GTSRB/val'

# Call the function
ppm_to_jpg_and_create_annotations(ppm_directory, jpg_train_directory,
                                  jpg_val_directory)
