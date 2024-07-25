from environment import *
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage

# Model
model = ChatOpenAI(model="gpt-4o", temperature=0.1)

# DataLoader
loader = TextLoader(file_path)
data = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
all_splits = text_splitter.split_documents(data)

# Embedding and store the data into vector db
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=5)
question = "Who is vincent jackson"
docs = retriever.invoke(question)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_TEMPLATE = """
Below are the file paths without punctuation. 
Retrieve all the file paths that may have the answer to the user's question. 
Provide only the paths as much as possible.

<web info>
{context}
</web info>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(model, question_answering_prompt)

# Prepare the input messages
messages = [
    HumanMessage(content=question),
]

# Execute the chain to get the response
response = document_chain.invoke({"messages": messages, "context": docs})

# Split the response into a list of lines
response_lines = response.split('\n')

# Print the resulting list
print(response_lines)