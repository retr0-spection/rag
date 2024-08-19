from app.utils.auth import auth_dependency
from fastapi import Body, APIRouter, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import Tuple, Annotated
#-----------------------------------
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from app.ingestion.utils import Ingestion

router = APIRouter(tags=["chat"])


GROQ_API = "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"

chats = {}

class Parameters(BaseModel):
    chat_id: str
    message: str


def fetch_customer_context(query_str: str, user_id: str):
    # In a real scenario, this could be a database query
    ingestion = Ingestion()
    ingestion.get_or_create_collection('embeddings')
    results = ingestion.query(query_str, user_id)
    return f"{results}"

@router.post('/chat')
def inference(parameters: Annotated[Parameters, Body()], user = Depends(auth_dependency)):
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API, model_name="llama-3.1-8b-instant")
    # Define a prompt template that uses the external context
    template = """Context: {context}

    Chat History: {chat_history}

    User Message: {message}
    AI Response: """

    # get history
    if parameters.chat_id not in chats:
        chats[parameters.chat_id] = ConversationBufferMemory(memory_key="chat_history")
    # Get the user's memory object
    memory = chats[parameters.chat_id]


    prompt = PromptTemplate(template=template, input_variables=["context","chat_history", "message"])

    # Create an LLMChain with OpenAI's GPT model
    llm_chain = prompt | llm

    context = fetch_customer_context(parameters.message, user.id)

    message = "I want to know more about the features of my new plan."


    response = llm_chain.invoke({"context":context, "message": parameters.message, "chat_history": memory.load_memory_variables({}).get("chat_history", "")})
    memory.save_context({"message": parameters.message}, {"AI": response.content})
    print(response.content)
    return {"message": response }
