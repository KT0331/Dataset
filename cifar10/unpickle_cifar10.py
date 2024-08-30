import numpy as np
import os
import cv2
import json

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

loc_0 = './cifar10'
loc_1 = os.path.join(loc_0, 'train')
loc_2 = os.path.join(loc_0, 'validation')
loc_3 = os.path.join(loc_0, 'test')

#判断文件夹是否存在，不存在的话创建文件夹
if os.path.exists(loc_0) == False:
    os.mkdir(loc_0)
if os.path.exists(loc_1) == False:
    os.mkdir(loc_1)
if os.path.exists(loc_2) == False:
    os.mkdir(loc_2)
if os.path.exists(loc_3) == False:
    os.mkdir(loc_3)


#训练集有五个批次，每个批次10000个图片, 將其中一個批次做為驗證集
#测试集有10000张图片
def cifar10_img(file_dir):
    
    # 訓練集
    filenames = []
    labels = []

    for i in range(1,5):
        data_name = file_dir + '/'+'data_batch_'+ str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')

        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j],(3,32,32))
            img = np.transpose(img,(1,2,0))
            #通道顺序为RGB
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #要改成不同的形式的文件只需要将文件后缀修改即可
            img_name = loc_1 + '/' + str(data_dict[b'labels'][j]) + str((i)*10000 + j) + '.jpg'
            cv2.imwrite(img_name,img)
            filenames.append(str(data_dict[b'labels'][j]) + str((i)*10000 + j) + '.jpg')
            labels.append(data_dict[b'labels'][j])

        print(data_name + ' is done')

    annotations = {
                    "filenames": filenames,
                    "labels": labels
                }

    # 指定要写入的文件名和路径
    file_path = os.path.join(loc_1, "annotations.json")

    # 将数据写入 JSON 文件
    with open(file_path, "w") as json_file:
        json.dump(annotations, json_file, indent=4)

    print('Train Set is Finished')

    # 驗證集
    filenames = []
    labels = []
    
    data_name = file_dir + '/'+'data_batch_'+ str(5)
    data_dict = unpickle(data_name)
    print(data_name + ' is processing')

    for j in range(10000):
        img = np.reshape(data_dict[b'data'][j],(3,32,32))
        img = np.transpose(img,(1,2,0))
        #通道顺序为RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #要改成不同的形式的文件只需要将文件后缀修改即可
        img_name = loc_2 + '/' + str(data_dict[b'labels'][j]) + str((5)*10000 + j) + '.jpg'
        cv2.imwrite(img_name,img)
        filenames.append(str(data_dict[b'labels'][j]) + str((5)*10000 + j) + '.jpg')
        labels.append(data_dict[b'labels'][j])

    print(data_name + ' is done')

    annotations = {
                    "filenames": filenames,
                    "labels": labels
                }

    # 指定要写入的文件名和路径
    file_path = os.path.join(loc_2, "annotations.json")

    # 将数据写入 JSON 文件
    with open(file_path, "w") as json_file:
        json.dump(annotations, json_file, indent=4)

    print('Validation Set is Finished')

    # 測試集
    filenames = []
    labels = []
    
    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)

    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        # 通道顺序为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 要改成不同的形式的文件只需要将文件后缀修改即可
        img_name = loc_3 + '/' + str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
        cv2.imwrite(img_name, img)
        filenames.append(str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg')
        labels.append(test_dict[b'labels'][m])
    print(test_data_name + ' is done')

    annotations = {
                    "filenames": filenames,
                    "labels": labels
                }

    # 指定要写入的文件名和路径
    file_path = os.path.join(loc_3, "annotations.json")

    # 将数据写入 JSON 文件
    with open(file_path, "w") as json_file:
        json.dump(annotations, json_file, indent=4)

    print('Test Set is Finished')

    print('Finish transforming to image')

    
if __name__ == '__main__':
    file_dir = './cifar-10-batches-py'
    cifar10_img(file_dir)
