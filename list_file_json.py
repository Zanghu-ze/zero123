import os
import json

def get_folder_names(base_directory, min_files_count):
    # 获取当前目录下所有符合文件数量要求的文件夹名
    folders = []
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path):
            # 获取该文件夹中的文件数量
            files_in_folder = os.listdir(folder_path)
            if len(files_in_folder) >= min_files_count:
                folders.append(folder_name)
    return folders

def save_to_json(data, output_file):
    # 将数据存储到JSON文件中
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # 当前文件夹路径
    base_directory = "/home/shared/haoze/indoor_data_partial"
    
    # 设置文件数量的筛选标准
    min_files_count = 8  # 要求每个文件夹至少有8个文件

    # 获取所有符合条件的文件夹
    folder_files = get_folder_names(base_directory, min_files_count)

    # 存储为JSON文件
    output_file = "valid_paths.json"
    save_to_json(folder_files, output_file)

    print(f"Filtered folder names saved to {output_file}")
