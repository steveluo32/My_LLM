import os
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader


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


def load_text(file_path):
    loader = TextLoader(file_path)
    data = loader.load()
    return data


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


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def load_word_document(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    data = loader.load()
    return data


def get_all_pdfs(directory):
    pdf_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

