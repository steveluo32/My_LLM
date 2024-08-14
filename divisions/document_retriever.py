import string
from config import *
from components.utils import find_key_by_value_content, load_text, load_content_dict, load_path_dict, get_all_pdfs, \
    load_pdf, get_all_file_paths, load_word_document, check_text_completeness
from components.model_initializer import chat_gpt, gemini, llama
from components.text_splitter import split_document_by_newline, recursive_character_splitter
from components.vectorstore_retriever import chroma_vectorstore, top_k_retriever, ensemble_retriever_2, bm25_retriever
from components.question_answering_chain import create_document_chain_document_retrieval, create_history, \
    execute_chain_without_memory, execute_chain_with_memory, create_document_chain_answer_giving_with_memory, \
    create_document_chain_answer_giving_without_memory
from components.vectorstore_retriever import historical_messages_retriever


class DocumentRetriever:
    def __init__(self):
        self.model = chat_gpt()
        # self.model = llama()
        # self.model = gemini()
        self.data = data_preparation()
        self.vectorstore = vector_store(self.data)
        self.retriever = create_retriever(self.data, self.vectorstore)
        # self.rag_chain = create_rag_chain_without_memory(self.model)
        self.rag_chain = create_document_chain_document_retrieval(self.model)
        self.chat_history = []

    def get_answer_without_memory(self, question):
        docs = self.retriever.invoke(question)
        answer = execute_chain_without_memory(self.rag_chain, question, docs)
        answer = check_text_completeness(self.model, answer)
        return answer

    def get_answer_with_memory(self, question):
        docs = self.retriever.invoke({"chat_history": self.chat_history, "input": question})
        answer = execute_chain_with_memory(self.rag_chain, question, docs, self.chat_history)
        answer = check_text_completeness(self.model, answer)
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


def load_data(path):
    docs = []
    for pdf_path in get_all_pdfs(path):
        docs.extend(load_pdf(pdf_path))
    for path in get_all_file_paths(path):
        if ".doc" in path or ".docx" in path:
            docs.extend(load_word_document(path))
        if ".txt" in path:
            docs.extend(load_text(path))

    all_splits = recursive_character_splitter(docs)
    return all_splits


def vector_store(data):
    vectorstore = chroma_vectorstore(data)
    return vectorstore


def data_preparation():
    data = load_text(clipped_path)
    split_data = split_document_by_newline(data)
    return split_data


def create_retriever(data, vectorstore):
    retriever = bm25_retriever(data, 50)
    # retriever = top_k_retriever(vectorstore, 20)
    # retriever = ensemble_retriever_2(retriever, data, 20)
    return retriever


def create_retriever_2(data, model, vectorstore):
    # retriever = bm25_retriever(all_splits, 100)
    retriever = top_k_retriever(vectorstore, 50)
    retriever = ensemble_retriever_2(retriever, data, 50)
    retriever = historical_messages_retriever(model, retriever)
    return retriever


def create_rag_chain_with_memory(model):
    # rag_chain = create_document_chain_answer_giving(model)
    rag_chain = create_document_chain_answer_giving_with_memory(model)
    return rag_chain


def create_rag_chain_without_memory(model):
    rag_chain = create_document_chain_answer_giving_without_memory(model)
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
    data = load_text(clipped_path)

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

