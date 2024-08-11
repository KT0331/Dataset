import pickle

# 載入 batches.meta 文件
with open('./cifar-10-batches-py/batches.meta', 'rb') as f:
    meta_data = pickle.load(f)

# 查看 batches.meta 文件中的內容
print('meta file:')
print(meta_data)

# 查看 label_names 鍵的值
print('label_names =', meta_data['label_names'])
