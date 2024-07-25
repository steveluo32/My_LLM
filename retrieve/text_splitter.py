from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(data, chunk_size=2000, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(data)
    return all_splits
