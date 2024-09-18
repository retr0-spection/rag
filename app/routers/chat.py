from app.models.contexthistory import ContextHistory
from app.utils.agents import setup_multi_agent_system
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict
from app.utils.auth import auth_dependency
from app.database import get_db, get_settings
from app.models.models import Agent, Session as ChatSession, Message
from app.ingestion.utils import Ingestion
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import AsyncIteratorCallbackHandler
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
import asyncio
from app.utils.agents import setup_multi_agent_system

router = APIRouter(tags=["chat"])

GROQ_API = get_settings().GROQ_API
FILE_MATCH_THRESHOLD = 0.7
RELEVANCE_THRESHOLD = 0.2

# Models for incoming data
class SendMessageParams(BaseModel):
    agent_id: int | None = None
    message: str
    session_id: Optional[int] = None

# Route to get chat sessions
@router.get('/chat/sessions')
def get_chat_sessions(db: Session = Depends(get_db), user = Depends(auth_dependency)):
    chat_sessions = db.query(ChatSession).filter_by(user_id=user.id).all()
    return {"chat_sessions": chat_sessions}

# Route to send a message (and start a new session if needed) with streaming response
@router.post('/chat')
async def send_message(params: SendMessageParams, db: Session = Depends(get_db), user = Depends(auth_dependency)):
    # If no session_id is provided, create a new chat session
    if params.session_id is None:
        # Check if agent exists
        agent = db.query(Agent).filter_by(id=params.agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Create a new chat session
        initial_title = " ".join(params.message.split()[:5]) + "..."
        chat_session = ChatSession(agent_id=params.agent_id, user_id=user.id, start_time=datetime.utcnow(), title=initial_title)
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        params.session_id = chat_session.id
    else:
        # Fetch the existing chat session
        chat_session = db.query(ChatSession).filter_by(id=params.session_id).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

    # Load the agent associated with the session
    agent = db.query(Agent).filter_by(id=chat_session.agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_config = json.loads(agent.configuration)

    multi_agent_system, callback_handler = await setup_multi_agent_system(db, user, agent_config, params.session_id)

    message_count = db.query(Message).filter_by(session_id=chat_session.id).count()
    # Update title every 5 messages
    if (message_count + 2) % 10 == 0:  # +2 because we just added two new messages
        messages = db.query(Message).filter_by(session_id=chat_session.id).order_by(Message.timestamp).all()
        llm = ChatGroq(temperature=0, groq_api_key=GROQ_API)
        new_title = await generate_ai_title(messages, llm)
        chat_session.title = new_title
        db.commit()
    async def generate_response():
        # Reattach the chat_session to the current database session
        db.add(chat_session)
        db.refresh(chat_session)
        full_response = ""

        async for token in callback_handler.aiter():
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        # # Save messages to database
        # user_message = Message(session_id=chat_session.id, sender='user', content=params.message, timestamp=datetime.utcnow())
        # ai_message = Message(session_id=chat_session.id, sender='AI', content=full_response, timestamp=datetime.utcnow())
        # db.add(user_message)
        # db.add(ai_message)
        # db.commit()



        yield f"data: {json.dumps({'session_id': chat_session.id, 'title': chat_session.title})}\n\n"
        yield "data: [DONE]\n\n"

    # Process the message with the multi-agent system
    asyncio.create_task(multi_agent_system.process_message({
        'message': params.message,
        'user_id': user.id,
        'system': agent_config['prompt'],
        'session_id': params.session_id
    }))

    return StreamingResponse(generate_response(), media_type="text/event-stream")

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


    return {"message": "Chat session ended"}

@router.delete('/chat/sessions/{session_id}')
def delete_chat_session(session_id: int, db: Session = Depends(get_db), user = Depends(auth_dependency)):
    # Fetch the chat session
    chat_session = db.query(ChatSession).filter_by(id=session_id, user_id=user.id).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Delete all messages associated with the session
    db.query(Message).filter_by(session_id=session_id).delete()

    # Delete all context associated with the session
    db.query(ContextHistory).filter_by(session_id=session_id).delete()

    # Delete the chat session
    db.delete(chat_session)
    db.commit()


    return {"message": "Chat session and associated messages deleted successfully"}

def rank_and_combine_results(file_matches: List[Dict], semantic_results: List[Dict]) -> List[Dict]:
    combined_results = file_matches + semantic_results
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    return combined_results


async def generate_ai_title(messages, llm):
    prompt = PromptTemplate(
        input_variables=["chat_history"],
        template="Summarize the following conversation in a short, concise title of 5-7 words(only provide the concise summary, nothing else):\n\n{chat_history}"
    )
    chat_history = "\n".join([f"{msg.sender}: {msg.content}" for msg in messages])
    response = await llm.apredict(prompt.format(chat_history=chat_history))
    return response.strip()
