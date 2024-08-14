import os

def get_all_files_in_subdirs(base_dir):
    file_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def save_file_paths_to_file(file_paths, file_path):
    with open(file_path, 'w') as f:
            for i, file_path in enumerate(file_paths):
                if (".DS_Store" not in file_path):
                    f.write(f"{i} {file_path}\n")

# 示例使用
base_directory = 'scraping_data'  # 将此处替换为你的基础目录路径
output_file = 'files_in_subdirs.txt'

all_files = get_all_files_in_subdirs(base_directory)
save_file_paths_to_file(all_files, output_file)

print(f"All file paths have been saved to {output_file}")
