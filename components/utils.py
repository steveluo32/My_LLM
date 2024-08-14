import os
import re
import time
import sys
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.prompts import PromptTemplate


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


def print_runtime(start_time):
    while True:
        current_time = time.time()
        runtime = current_time - start_time
        sys.stdout.write(f"\rRuntime: {runtime:.2f} seconds")
        sys.stdout.flush()
        time.sleep(1)


def check_text_completeness(model, text):
    # Define the prompt template
    template = """
    You are a text-checking assistant. Given a piece of text, please tell me if the end of the text is complete.
    - If the text is incomplete, complete it and return ONLY the part of the text that was incomplete.
    - If it's complete, return "The text is complete."
    Text: "{text}"
    Complete it if incomplete
    """
    prompt = PromptTemplate(input_variables=["text"], template=template)
    # Create the chain
    completeness_chain = prompt | model
    response = completeness_chain.invoke({"text": text})
    pattern = r'"(.*?)"'

    if "The text is complete" in response.content or "the text is complete" in response.content:
        return text
    elif re.search(pattern, response.content):
        last_word = re.search(pattern, response.content).group(1)
        text = remove_repetition_in_last_sentence(text + " " +last_word)
        return text
    else:
        text = text + " " + response.content
        text = remove_repetition_in_last_sentence(text)
        return text


def remove_repetition_in_last_sentence(sentence: str) -> str:
    # Step 1: Split the sentence into parts while keeping the delimiters (punctuation)
    sentences = re.split(r'([.!?。！？]\s*)', sentence.strip())

    # Step 2: Extract the last sentence
    if len(sentences) == 1:
        last_sentence = sentences[0]
    elif sentences[-1] == '':
        last_sentence = sentences[-3] + sentences[-2]
    else:
        last_sentence = sentences[-1]

    last_sentence = re.sub(r'\s+', ' ', last_sentence)

    # Step 3: Use regex to identify and remove repetition in the last sentence
    pattern = r'\b(\w+(?:\s+\w+)*)\s+\1\b'
    processed_sentence = re.sub(pattern, r'\1', last_sentence.lower())

    # Step 4: Capitalize the first letter of the processed sentence
    processed_sentence = processed_sentence.capitalize()

    # Step 5: Reassemble the sentence
    if len(sentences) == 1:
        result_sentence = processed_sentence
    elif sentences[-1] == '':
        result_sentence = ''.join(sentences[:-3]) + processed_sentence
    else:
        result_sentence = ''.join(sentences[:-1]) + processed_sentence

    return result_sentence

