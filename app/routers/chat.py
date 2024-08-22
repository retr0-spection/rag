from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from app.utils.auth import auth_dependency
from app.database import get_db
from app.models.models import Agent, Session as ChatSession, Message
from app.ingestion.utils import Ingestion
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import json

router = APIRouter(tags=["chat"])

GROQ_API = "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"
chats = {}

# Models for incoming data
class StartSessionParams(BaseModel):
    agent_id: int

class SendMessageParams(BaseModel):
    session_id: int
    message: str

# Route to start a chat session
@router.post('/chat/sessions')
def start_chat_session(params: StartSessionParams, db: Session = Depends(get_db), user = Depends(auth_dependency)):
    # Check if agent exists
    agent = db.query(Agent).filter_by(id=params.agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Create a new chat session
    chat_session = ChatSession(agent_id=params.agent_id, user_id=user.id, start_time=datetime.utcnow())
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)

    # Initialize memory for the session
    chats[chat_session.id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return {"session_id": chat_session.id}

# Route to send a message
@router.post('/chat/messages')
def send_message(params: SendMessageParams, db: Session = Depends(get_db), user = Depends(auth_dependency)):
    # Fetch the chat session
    chat_session = db.query(ChatSession).filter_by(id=params.session_id).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Load the agent associated with the session
    agent = db.query(Agent).filter_by(id=chat_session.agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")


    agent_config = json.loads(agent.configuration)

    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API, model_name=agent_config['llm'])

    template = """
    System: {system}

    Context: {context}

    Chat History: {chat_history}

    User Message: {message}
    AI Response: """

    if params.session_id not in chats:
        chats[params.session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Get the user's memory object
    memory = chats[params.session_id]
    prompt = PromptTemplate(template=template, input_variables=["system","context", "chat_history", "message"])
    context = fetch_customer_context(params.message, user.id)

    # Generate the AI response
    llm_chain = prompt | llm
    response = llm_chain.invoke({
        "system":agent_config['prompt'],
        "context": context,
        "message": params.message,
        "chat_history": memory.load_memory_variables({}).get("chat_history", "")
    })

    # Save message and response to database
    user_message = Message(
        session_id=chat_session.id,
        sender='user',
        content=params.message,
        timestamp=datetime.utcnow()
    )
    ai_message = Message(
        session_id=chat_session.id,
        sender='AI',
        content=response.content,
        timestamp=datetime.utcnow()
    )

    db.add(user_message)
    db.add(ai_message)
    db.commit()

    # Save the conversation to the in-memory chat history
    memory.save_context({"message": params.message}, {"AI": response.content})

    return {"message": response.content}

# Route to retrieve chat history
@router.get('/chat/sessions/{session_id}/history')
def get_chat_history(session_id: int, db: Session = Depends(get_db)):
    # Fetch the chat session
    chat_session = db.query(ChatSession).filter_by(id=session_id).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Fetch all messages in the session
    messages = db.query(Message).filter_by(session_id=session_id).order_by(Message.timestamp).all()

    return {
        "session_id": session_id,
        "messages": [{"sender": msg.sender, "content": msg.content, "timestamp": msg.timestamp} for msg in messages]
    }

# Route to end a chat session
@router.post('/chat/sessions/{session_id}/end')
def end_chat_session(session_id: int, db: Session = Depends(get_db)):
    # Fetch the chat session
    chat_session = db.query(ChatSession).filter_by(id=session_id).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Mark the session as ended
    chat_session.end_time = datetime.utcnow()
    db.commit()

    # Optionally, clear the in-memory chat history
    if session_id in chats:
        del chats[session_id]

    return {"message": "Chat session ended"}

# Utility function to fetch customer context
def fetch_customer_context(query_str: str, user_id: str):
    ingestion = Ingestion()
    ingestion.get_or_create_collection('embeddings')
    results = ingestion.query(query_str, user_id)
    print(results)
    return f"{results}"
