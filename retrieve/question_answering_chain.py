from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

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

def create_document_chain(model):
    document_chain = create_stuff_documents_chain(model, question_answering_prompt)
    return document_chain

def execute_chain(document_chain, question, context):
    messages = [
        HumanMessage(content=question),
    ]
    response = document_chain.invoke({"messages": messages, "context": context})
    return response
