from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_google_vertexai import VertexAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def chat_gpt(model_name="gpt-4o", temperature=0.01):
    model = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=3000)
    return model


def gemini(model_name="gemini-1.5-pro", temperature=0.01):
    model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    return model


def vertex(model_name="gemini-1.5-pro", temperature=0.01):
    model = VertexAI(model_name=model_name, temperature=temperature)
    return model


def llama_3_1(model_name="meta-llama/Meta-Llama-3.1-8B"):
    model_id = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


if __name__ == "__main__":
    llama_3_1()
