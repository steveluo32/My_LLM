import os
from key import *

os.environ["OPENAI_API_KEY"] = openai.api_key

data_path = "./data"
clipped_path = "./data/clipped_path.txt"
real_path = "./data/real_path.txt"
llama_path = "./Meta-Llama-3.1-8B"