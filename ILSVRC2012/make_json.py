import json

with open('../annotations.json', 'r') as file:
    data = json.load(file)

# 提取 filenames 项目，并删除每个字符串的前 10 个字符
filenames = data["filenames"]
modified_filenames = [filename[10:] for filename in filenames]

# 更新 JSON 数据中的 filenames 项目
data["filenames"] = modified_filenames

# 输出修改后的 JSON 数据
with open('annotations.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
