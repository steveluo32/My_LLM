from langchain_community.document_loaders.text import TextLoader

def load_data(file_path):
    loader = TextLoader(file_path)
    data = loader.load()
    return data
