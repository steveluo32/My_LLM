import os
import re

def get_all_files_in_subdirs(base_dir):
    file_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def clean_path(path):
    # 移除路径中的所有符号，只保留字母和数字
    return re.sub(r'[^a-zA-Z0-9]', ' ', path)


def save_file_paths_to_file(file_paths, file_path):
    with open(file_path, 'w') as f:
        for i, file_path in enumerate(file_paths):
            cleaned_path = clean_path(file_path)
            f.write(f"{i} {cleaned_path}\n")


# 示例使用
base_directory = 'scraping_data'  # 将此处替换为你的基础目录路径
output_file = 'files_in_subdirs2.txt'

all_files = get_all_files_in_subdirs(base_directory)
save_file_paths_to_file(all_files, output_file)

print(f"All file paths have been saved to {output_file}")