import pickle

# 載入 batches.meta 文件
with open('./cifar-100-python/meta', 'rb') as f:
    meta_data = pickle.load(f)

# 查看 batches.meta 文件中的內容
print('meta file:')
fine_label_names = meta_data['fine_label_names']
coarse_label_names = meta_data['coarse_label_names']

print(f'fine_label_names:\n{fine_label_names}')
print(f'coarse_label_names:\n{coarse_label_names}')
