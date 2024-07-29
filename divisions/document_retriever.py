import string
from config import *
from components.utils import find_key_by_value_content, load_data, load_content_dict, load_path_dict
from components.model_initializer import chat_gpt, llama_3_1
from components.text_splitter import split_document_by_newline
from components.vectorstore_retriever import chroma_vectorstore, top_k_retriever, ensemble_retriever_2, bm25_retriever
from components.question_answering_chain import create_document_chain_document_retrieval, create_history, \
    execute_chain_without_memory, execute_chain_with_memory, create_document_chain_with_memory, \
    create_document_chain_answer_giving
from components.vectorstore_retriever import historical_messages_retriever

class DocumentRetriever:
    def __init__(self):
        self.model = chat_gpt()
        self.data = data_preparation()
        self.vectorstore = vector_store(self.data)
        self.retriever = create_retriever(self.data, self.vectorstore)
        # self.rag_chain = create_rag_chain_without_memory(self.model)
        self.rag_chain = create_document_chain_document_retrieval(self.model)
        self.chat_history = []

    def get_answer_without_memory(self, question):
        docs = self.retriever.invoke(question)
        response = execute_chain_without_memory(self.rag_chain, question, docs)
        return response

    def get_answer_with_memory(self, question):
        docs = self.retriever.invoke({"chat_history": self.chat_history, "input": question})
        answer = execute_chain_with_memory(self.rag_chain, question, docs)
        new_history = create_history(question, answer)
        self.chat_history.extend(new_history)
        return answer

    def retrieve_document(self, question):
        response = self.get_answer_without_memory(question)
        # Split and print the response
        retrieve_paths = response.split('\n')
        # process to produce real paths
        paths = process_retrieved_paths(retrieve_paths)
        return paths

def vector_store(data):
    vectorstore = chroma_vectorstore(data)
    return vectorstore

def data_preparation():
    data = load_data(clipped_path)
    split_data = split_document_by_newline(data)
    return split_data

def create_retriever(data, vectorstore):
    # retriever = bm25_retriever(data, 100)
    retriever = top_k_retriever(vectorstore, 20)
    retriever = ensemble_retriever_2(retriever, data, 20)
    return retriever

def create_retriever_2(data, model, vectorstore):
    # retriever = bm25_retriever(all_splits, 100)
    retriever = top_k_retriever(vectorstore, 20)
    retriever = ensemble_retriever_2(retriever, data, 20)
    retriever = historical_messages_retriever(model, retriever)
    return retriever

def create_rag_chain_with_memory(model):
    # rag_chain = create_document_chain_answer_giving(model)
    rag_chain = create_document_chain_with_memory(model)
    return rag_chain

def create_rag_chain_without_memory(model):
    rag_chain = create_document_chain_answer_giving(model)
    return rag_chain

def process_retrieved_paths(retrieved_paths):
    # Load dictionaries
    content_dict = load_content_dict(clipped_path)
    path_dict = load_path_dict(real_path)

    # Initialize lists
    num_list = []
    path_list = []

    # Find keys by value content and populate num_list
    for line in retrieved_paths:
        line = line.translate(str.maketrans('', '', string.punctuation+string.digits)).strip()
        num = find_key_by_value_content(content_dict, line)
        if num:
            num_list.append(num)

    # Populate path_list with corresponding paths
    for num in num_list:
        path = os.path.join(data_path, path_dict[num])
        path_list.append(path)

    return path_list

@DeprecationWarning
def document_retriever(question):
    # Initialize model
    model = chat_gpt()

    # Load data
    data = load_data(clipped_path)

    # Split text into chunks
    # all_splits = recursive_character_splitter(data)
    all_splits = split_document_by_newline(data)
    # all_splits = semantic_splitter(data)

    # Create vector store and retriever
    vectorstore = chroma_vectorstore(all_splits)
    retriever = bm25_retriever(all_splits, 20)
    # retriever = top_k_retriever(vectorstore, 100)
    # retriever = ensemble_retriever_2(retriever, data, 100)

    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Create and execute question answering chain
    document_chain = create_document_chain_document_retrieval(model)
    response = execute_chain_without_memory(document_chain, question, docs)

    # Split and print the response
    retrieve_paths = response.split('\n')

    return retrieve_paths

def main_retrieve(question):
    retrieve_paths = document_retriever(question)
    paths = process_retrieved_paths(retrieve_paths)
    return paths
