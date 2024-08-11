import os
import tarfile

import os
import tarfile

def extract_tar_files_in_directory(directory):
    """Extracts all tar files in the specified directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".tar"):
            tar_path = os.path.join(directory, filename)
            extract_path = os.path.join(directory, filename.split('.')[0])
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_path)
                os.remove(tar_path)
                # print(f"Extracted {tar_path} to {extract_path}")

def extract_tar_file(tar_path, extract_path):
    """Extracts a tar file to a specified directory."""
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)
        print(f"Extracted {tar_path} to {extract_path}")

# Setting Path
# 训练集 tar 文件路径
train_tar_path = './imagenet/ILSVRC2012_img_train.tar'
# 验证集 tar 文件路径
val_tar_path = './imagenet/ILSVRC2012_img_val.tar'
# Devkit tar 文件路径
devkit_tar_path = './imagenet/ILSVRC2012_devkit_t12.tar.gz'
# 训练集解压目录
train_extract_path = './ILSVRC2012/train'
# 验证集解压目录
val_extract_path = './ILSVRC2012/val'
 # Devkit 解压目录
devkit_extract_path = './ILSVRC2012/'

# 创建目标目录
os.makedirs(train_extract_path, exist_ok=True)
os.makedirs(val_extract_path, exist_ok=True)

# 解压训练集和验证集
print(f'Unzip Train Image Start...')
extract_tar_file(train_tar_path, train_extract_path)
print(f'Unzip Train Image End...')
print(f'Unzip Val Image Start...')
extract_tar_file(val_tar_path, val_extract_path)
print(f'Unzip Val Image End...')
print(f'Unzip Train Image Class Start...')
extract_tar_files_in_directory(train_extract_path)
print(f'Unzip Train Image Class End...')
# print(f'Unzip devkit Start...')
# extract_tar_file(devkit_tar_path, devkit_extract_path)
# print(f'Unzip devkit End...')