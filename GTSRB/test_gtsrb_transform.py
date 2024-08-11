import pandas as pd
import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageNet

test_ppm_path = './GTSRB/Final_Test/Images/'
test_target_dir = './test'

os.makedirs(test_target_dir, exist_ok=True)

print(f"Read CSV Start...")

# 读取 CSV 文件
df = pd.read_csv('./GT-final_test.csv', delimiter=';')

# 提取 Filename 和 ClassId 列
filenames = df['Filename'].tolist()
class_ids = df['ClassId'].tolist()

print(f"Read CSV Complete...")

print(f"Image Conversion Start...")

# 输出 Filename 和 ClassId 列
for filename, class_id in zip(filenames, class_ids):

    ppm_file = os.path.join(test_ppm_path, filename)
    jpg_filename = filename.replace('.ppm', '.jpg')
    jpg_path = os.path.join(test_target_dir, jpg_filename)
    with Image.open(ppm_file) as img:
            img.save(jpg_path, 'JPEG')

print(f"Image Conversion Complete...")

annotations_test = {
                    "filenames": jpg_filename,
                    "labels": class_ids
                   }

with open(os.path.join(test_target_dir, 'annotations.json'), 'w') as json_file:
    json.dump(annotations_test, json_file, indent=4)

print(f'Test Annotation Finished...')
print(f'Test Set Conversion Complete...')
