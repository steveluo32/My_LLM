from answer.model_initializer import initialize_model
from answer.file_loader import load_content_dict, load_path_dict
from answer.utils import find_key_by_value_content, read_files
from langchain_core.messages import HumanMessage
from answer.answer_environment import *

def process_retrieved_paths(retrieved_paths):
    # Load dictionaries
    content_dict = load_content_dict(clipped_path)
    path_dict = load_path_dict(real_path)

    # Initialize lists
    num_list = []
    path_list = []

    # Find keys by value content and populate num_list
    for line in retrieved_paths:
        num = find_key_by_value_content(content_dict, line)
        if num:
            num_list.append(num)

    # Populate path_list with corresponding paths
    for num in num_list:
        path = os.path.join(data_path, path_dict[num])
        path_list.append(path)

    return path_list

def main_answer(question, path_list):
    # Initialize model
    chat = initialize_model()

    # Check if path_list is not empty and process accordingly
    if path_list:
        file_content = read_files(path_list)
        messages = [HumanMessage(content=file_content + "\n" + question)]
    else:
        messages = [HumanMessage(content=question)]

    # Get response from the chat model
    response = chat.invoke(messages)

    return response.content


