from langchain_openai import ChatOpenAI
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def chat_gpt(model_name="gpt-4o", temperature=0.01):
    model = ChatOpenAI(model=model_name, temperature=temperature)
    return model


def llama_3_1():
    model_id = "Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

