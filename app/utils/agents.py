import abc
from app.models.contexthistory import ContextHistory
from fastapi import HTTPException
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from app.ingestion.utils import Ingestion
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict
from app.utils.auth import auth_dependency
from app.database import get_db
from app.models.models import Agent, Session as ChatSession, Message
from app.ingestion.utils import Ingestion
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import AsyncIteratorCallbackHandler
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
import asyncio
from langchain_core.messages import AIMessage
from langchain.schema import HumanMessage


GROQ_API = "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"

FILE_MATCH_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 0.15

class Agent(abc.ABC):
    @abc.abstractmethod
    async def process_task(self, task: Dict) -> str:
        pass

class RouterAgent(Agent):
    def __init__(self, specialized_agents: List[Agent]):
        self.specialized_agents = specialized_agents

    async def process_task(self, task: Dict) -> str:
        appropriate_agent = self.select_agent(task)
        return await appropriate_agent.process_task(task)

    def select_agent(self, task: Dict) -> Agent:
        # Implement logic to select the appropriate agent based on the task
        # For simplicity, we'll just return the first agent here
        return self.specialized_agents[0]

class ContextAgent(Agent):
    def __init__(self, ingestion: Ingestion):
        self.ingestion = ingestion

    async def process_task(self, task: Dict) -> str:
        query = task['message']
        user_id = task['user_id']
        if needs_context(query):
            return fetch_customer_context(query, user_id)
        return ""

class LLMAgent(Agent):
    def __init__(self, llm: ChatGroq, prompt_template: PromptTemplate, memory: ConversationBufferMemory,session_id, db: Session):
        self.llm = llm
        self.prompt_template = prompt_template
        self.memory = memory
        self.db = db
        self.session_id = session_id

    async def process_task(self, task: Dict) -> str:
        new_context = task.get('context', '')
        message = task['message']
        # Get updated chat history

        # Load previous messages into memory
        self.load_previous_messages(self.session_id)

        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")



        # Store new context if provided
        if new_context:
            self.store_context(self.session_id, new_context)

        # Retrieve context history
        context_history = self.get_context_history(self.session_id)


        # Combine context history
        context = "\n".join(context_history)

        prompt = self.prompt_template.format(
            system=task['system'],
            context=context,
            chat_history=chat_history,
            message=message
        )

        print(prompt)

        try:
            response = await self.llm.ainvoke(prompt)

            if isinstance(response, AIMessage):
                response_content = response.content
            elif isinstance(response, str):
                response_content = response
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")

            self.memory.save_context({"input": message}, {"output": response_content})
            self.store_message(self.session_id, message, sender="user")
            self.store_message(self.session_id, response_content, sender="AI")
            return response_content

        except Exception as e:
            print(f"Error in LLMAgent: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def store_context(self, session_id: int, context: str):
        new_context = ContextHistory(session_id=session_id, context=context)
        self.db.add(new_context)
        self.db.commit()

    def get_context_history(self, session_id: int, limit: int = 3) -> List[str]:
        context_history = self.db.query(ContextHistory) \
            .filter(ContextHistory.session_id == session_id) \
            .order_by(ContextHistory.created_at.desc()) \
            .limit(limit) \
            .all()
        return [ch.context for ch in reversed(context_history)]

    def load_previous_messages(self, session_id: int):
        messages = self.db.query(Message) \
            .filter(Message.session_id == session_id) \
            .order_by(Message.timestamp.asc()) \
            .all()

        for msg in messages:
            if msg.sender == "user":
                self.memory.chat_memory.add_user_message(msg.content)
            else:
                self.memory.chat_memory.add_ai_message(msg.content)

    def store_message(self, session_id: int, content: str, sender: str):
        new_message = Message(session_id=session_id, content=content, sender=sender)
        self.db.add(new_message)
        self.db.commit()


class MultiAgentSystem:
    def __init__(self, router: RouterAgent, context_agent: ContextAgent, llm_agent: LLMAgent):
        self.router = router
        self.context_agent = context_agent
        self.llm_agent = llm_agent

    async def process_message(self, params: Dict) -> str:
        try:
            # Get context if needed
            context = await self.context_agent.process_task(params)
            params['context'] = context

            # Process the message with the LLM agent
            response = await self.llm_agent.process_task(params)

            return response
        except Exception as e:
            print(f"Error in MultiAgentSystem: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."


def fetch_customer_context(query_str: str, user_id: str):
    ingestion = Ingestion()
    ingestion.get_or_create_collection('embeddings')

    # First, try to find file matches
    file_matches = ingestion.query_file_names(query_str, user_id)

    # If we have strong file matches, return those
    if len(file_matches) and file_matches[0]['similarity'] > FILE_MATCH_THRESHOLD:
        return "\n".join([match['text'] for match in file_matches])

    # Otherwise, perform semantic search
    semantic_results = ingestion.query(query_str, user_id)

    # Combine and rank results
    combined_results = rank_and_combine_results(file_matches, semantic_results)

    return "\n".join([result['text'] for result in combined_results])

def needs_context(query: str) -> bool:
    # Convert query to lowercase
    query = query.lower()

    # Tokenize the query
    tokens = word_tokenize(query)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Define context-indicating keywords
    context_keywords = [
        'background', 'history', 'details', 'information', 'context',
        'explain', 'elaborate', 'clarify', 'describe', 'summarize',
        'overview', 'introduction', 'summary', 'breakdown', 'analysis',
        'file', 'document'  # Add keywords related to file requests
    ]

    # Check for presence of context-indicating keywords
    if any(keyword in tokens for keyword in context_keywords):
        return True

    # Check for question words that often require context
    question_words = ['who', 'what', 'where', 'when', 'why', 'how']
    if any(word in tokens for word in question_words):
        return True

    # Check for comparative or superlative adjectives
    if any(token.endswith(('er', 'est')) for token in tokens):
        return True

    # Check for phrases indicating a need for context or file request
    context_phrases = [
        'tell me about', 'give me information on', 'what do you know about',
        'can you explain', 'i need to understand', 'provide details on',
        'show me the file', 'find the document', 'search for the file'  # Add phrases related to file requests
    ]
    if any(phrase in query for phrase in context_phrases):
        return True

    # Check query length - longer queries might need context
    if len(tokens) > 10:
        return True

    # Check for presence of proper nouns (potential named entities)
    if any(token.istitle() for token in tokens):
        return True

    # If none of the above conditions are met, context might not be needed
    return False

def rank_and_combine_results(file_matches: List[Dict], semantic_results: List[Dict]) -> List[Dict]:
    combined_results = file_matches + semantic_results
    print(combined_results)
    combined_results.sort(key=lambda x: x['similarity'], reverse=True)
    return combined_results

async def setup_multi_agent_system(db, user, agent_config, session_id):
    ingestion = Ingestion()
    context_agent = ContextAgent(ingestion)

    callback_handler = AsyncIteratorCallbackHandler()
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API,
        model_name=agent_config['llm'],
        streaming=True,
        callbacks=[callback_handler]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = PromptTemplate(
        template="""

        System: {system}.
        Resource: {context}
        Chat History: {chat_history}
        User Message: {message}
        Assistant: Output is used in ReactMarkdown component. Format your responses nicely. Don't be afraid to use newlines and space.
        AI Response:""",
        input_variables=["system", "context", "chat_history", "message"]
    )

    llm_agent = LLMAgent(llm, prompt_template, memory, session_id, db)

    router = RouterAgent([context_agent, llm_agent])

    return MultiAgentSystem(router, context_agent, llm_agent), callback_handler
