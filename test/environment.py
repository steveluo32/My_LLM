import os

import openai

openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = openai.api_key

data_path = "./data"
file_path = "./data/file.txt"
path = "./data/path_file.txt"