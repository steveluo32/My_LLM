def load_content_dict(file_path):
    content_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            key, value = line.split(' ', 1)
            content_dict[key] = line
    return content_dict

def load_path_dict(path):
    path_dict = {}
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            key, value = line.split(' ', 1)
            path_dict[key] = value
    return path_dict
