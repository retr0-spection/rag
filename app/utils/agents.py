import abc
from typing import Dict, List, Tuple, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from pydantic import BaseModel, Field

from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from app.ingestion.utils import Ingestion
from app.database import get_db, get_settings
from app.models.contexthistory import ContextHistory
from app.models.models import Agent, Session as ChatSession, Message
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
import asyncio

# Constants
GROQ_API = get_settings().GROQ_API
FILE_MATCH_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 0.15
TAG_MATCH_THRESHOLD = 0.8

# Existing utility functions
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
        'file', 'document', 'data'
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
        'show me the file', 'find the document', 'search for the file'
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

async def fetch_customer_context(query_str: str, user_id: str):
    ingestion = Ingestion()
    ingestion.get_or_create_collection('embeddings')

    # First, try to find file matches
    file_matches = ingestion.query_file_names(query_str, user_id)

    # Then, try to find tag matches
    tag_matches = await ingestion.query_by_tags(query_str, user_id)

    # If we have strong file matches, prioritize those
    if len(file_matches) and file_matches[0]['similarity'] > FILE_MATCH_THRESHOLD:
        return "\n".join([match['text'] for match in file_matches])

    # If we have strong tag matches, include those
    tag_matched_files = [match for match in tag_matches if match['combined_score'] > TAG_MATCH_THRESHOLD]

    # Perform semantic search
    semantic_results = await ingestion.query(query_str, user_id, relevance_threshold=RELEVANCE_THRESHOLD)

    # Combine and rank results
    combined_results = rank_and_combine_results(file_matches, tag_matched_files, semantic_results)

    # Extract and combine relevant text
    context_text = []
    for result in combined_results:
        if 'text' in result:
            context_text.append(result['text'])
        elif 'sample_text' in result:
            context_text.append(result['sample_text'])

    return "\n".join(context_text)

def rank_and_combine_results(file_matches: List[Dict], tag_matches: List[Dict], semantic_results: List[Dict]) -> List[Dict]:
    combined_results = []

    # Process file matches
    for match in file_matches:
        combined_results.append({
            'text': match['text'],
            'score': match['similarity'],
            'type': 'file_match'
        })

    # Process tag matches
    for match in tag_matches:
        combined_results.append({
            'sample_text': match.get('sample_text', ''),
            'score': match['combined_score'],
            'type': 'tag_match',
            'matching_tags': match['matching_tags']
        })

    # Process semantic results
    for result in semantic_results:
        combined_results.append({
            'text': result['text'],
            'score': result['similarity'],
            'type': 'semantic_match'
        })

    # Sort combined results by score in descending order
    combined_results.sort(key=lambda x: x['score'], reverse=True)

    return combined_results

# New LangGraph implementation

class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: str
    user_id: str
    session_id: int

class ContextTool(BaseModel):
    name: str = "fetch_context"
    description: str = "Fetch context for a given query and user ID"
    query: str = Field(description="The query to fetch context for")
    user_id: str = Field(description="The user ID to fetch context for")

    async def __call__(self, query: str, user_id: str) -> str:
        return await fetch_customer_context(query, user_id)

class LLMNode:
    def __init__(self, llm: ChatGroq, prompt: ChatPromptTemplate, memory: ConversationBufferMemory, db):
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
        self.db = db

    def __call__(self, state: AgentState) -> Dict:
        messages = state['messages']
        context = state['context']
        user_message = messages[-1].content if messages else ""

        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")

        full_prompt = self.prompt.format(
            system="You are a helpful assistant.",
            context=context,
            chat_history=chat_history,
            message=user_message
        )

        ai_message = self.llm.invoke(full_prompt)

        # Store the message
        self.store_message(state['session_id'], user_message, "user")
        self.store_message(state['session_id'], ai_message.content, "AI")

        # Update memory
        self.memory.save_context({"input": user_message}, {"output": ai_message.content})

        return {
            "messages": messages + [AIMessage(content=ai_message.content)],
            "context": context,
            "user_id": state['user_id'],
            "session_id": state['session_id']
        }




    def store_message(self, session_id: int, content: str, sender: str):
        new_message = Message(session_id=session_id, content=content, sender=sender)
        self.db.add(new_message)
        self.db.commit()

def context_agent(state: AgentState) -> Tuple[AgentState, str]:
    user_message = state['messages'][-1].content
    # if needs_context(user_message):
    #     return state, "fetch_context"
    return state, "llm"

def router(state: AgentState) -> Dict:
    _dict = {
        "messages":state['messages'],
        "context":state["context"],
        "user_id":state['user_id'],
        "session_id":state['session_id'],
        "end":True
    }
    return _dict

async def setup_langgraph_system(user, agent_config, session_id, db):
    llm = ChatGroq(
        temperature=0.1,
        groq_api_key=GROQ_API,
        model_name=agent_config['llm'],
        streaming=True
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    restore_memory_state(memory, session_id, db)

    print(agent_config)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"System: {agent_config['prompt']}. If you're given context, always try to see if it is relevant to answering the question otherwise ignore it. Consult the resource given below when appropriate."),
        ("system", "Resource: {context}"),
        ("system", "Chat History: {chat_history}"),
        ("human", "{message}"),
    ])

    llm_node = LLMNode(llm, prompt, memory, db)

    # tools = [ContextTool]
    # tool_executor = ToolExecutor(tools)

    workflow = StateGraph(AgentState)

    # workflow.add_node("context_agent", context_agent)
    # workflow.add_node("fetch_context", tool_executor)
    workflow.add_node("llm", llm_node)
    workflow.add_node("router", router)

    workflow.set_entry_point("llm")

    # workflow.add_edge("context_agent", "fetch_context")
    # workflow.add_edge("context_agent", "llm")
    # workflow.add_edge("fetch_context", "llm")
    workflow.add_edge("llm", "router")
    workflow.add_edge("router", END)

    app = workflow.compile()

    return app

# Usage function
async def process_message(sys, message: str, user_id: str, session_id: int):
    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        context="",
        user_id=user_id,
        session_id=session_id
    )

    async for chunk in sys.astream(initial_state):
        print(chunk)
        try:
            yield "__token__:" + chunk['router']['messages'][-1].content
        except Exception:
            pass

def restore_memory_state(memory, session_id, db):
    messages = db.query(Message).filter_by(session_id=session_id).order_by(Message.timestamp)

    for message in messages:
        if message.sender == "user":
            memory.save_context({"input": message.content},{"output": ""})
        else:
            memory.save_context({"input": ""},{"output": message.content})
