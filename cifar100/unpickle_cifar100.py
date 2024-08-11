import os
import pickle
import numpy as np
from PIL import Image
import json
from sklearn.model_selection import train_test_split

# CIFAR-100 数据集路径
cifar100_dir = './cifar-100-python'

# 读取数据
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# 创建文件夹
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 保存图片并生成 JSON
def save_images_and_generate_json(data, labels, label_names, split_name, output_dir):
    images_dir = os.path.join(output_dir, split_name)
    create_dir(images_dir)
    
    filenames = []
    labelnames = []
    
    for i, (img, label) in enumerate(zip(data, labels)):
        # label_name = label_names[label]
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        
        img_filename = f'{split_name}_{i:05d}.png'
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)

        filenames.append(img_filename)
        # labelnames.append(label_name)
        labelnames.append(label)

    annotations = {
                    "filenames": filenames,
                    "labels": labelnames
                  }
    
    json_path = os.path.join(images_dir, "annotations.json")

    with open(json_path, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)

# 主程序
def main():
    # 加载元数据
    meta = unpickle(os.path.join(cifar100_dir, 'meta'))
    fine_label_names = meta['fine_label_names']
    
    # 加载训练数据
    train_data_dict = unpickle(os.path.join(cifar100_dir, 'train'))
    train_data = train_data_dict['data']
    train_labels = train_data_dict['fine_labels']
    
    # 加载测试数据
    test_data_dict = unpickle(os.path.join(cifar100_dir, 'test'))
    test_data = test_data_dict['data']
    test_labels = test_data_dict['fine_labels']
    
    # 将训练数据分为训练集和验证集
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    
    # 输出目录
    output_dir = '../cifar100'
    create_dir(output_dir)
    
    # 保存图片和生成 JSON
    save_images_and_generate_json(train_data, train_labels, fine_label_names, 'train', output_dir)
    save_images_and_generate_json(val_data, val_labels, fine_label_names, 'val', output_dir)
    save_images_and_generate_json(test_data, test_labels, fine_label_names, 'test', output_dir)
    
    print("Images and JSON files have been successfully saved!")

if __name__ == "__main__":
    main()
