import os

def find_key_by_value_content(dictionary, search_str):
    for key, value in dictionary.items():
        if search_str in value:
            return key
    return None

def read_files(file_paths):
    content = ""
    for path in file_paths:
        with open(path, 'r', errors='ignore') as file:
            content += file.read() + "\n"
    return content
