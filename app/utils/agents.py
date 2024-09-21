import abc
from typing import Dict, List, Tuple, Annotated, TypedDict, Literal
from app.models.file import File
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolInvocation, ToolExecutor, ToolNode
from pydantic import BaseModel, Field
from langchain_core.tools import tool
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
import functools

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


@tool
def get_document_contents(document_name:Annotated[str, "Document name as seen in knowledge base"], user_id:Annotated[str, "user id"]):
    """
    Retrieve all chunks for a specific document name and user ID.

    Args:
        document_name (str): TDocument name as seen in knowledge base.
        user_id (str): The ID of the user who owns the document.

    Returns:
        List[Dict]: A list of dictionaries, each containing a chunk's content and metadata.
    """

    results = Ingestion().get_document_chunks(document_name, user_id)
    print(results)
    return results

@tool
async def fetch_customer_context(query_str: Annotated[str, "User prompt"], user_id: Annotated[str, "user id"]) -> str:
    '''Fetch internal resources and user documents that might be relevant to the query.'''
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


tools = [fetch_customer_context, get_document_contents]
tool_node = ToolNode(tools)
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
    user_id: str
    session_id: int
    sender: str
    current_agent: str  # New field to track the current agent

class LLMNode:
    def __init__(self, llm: ChatGroq, prompt: ChatPromptTemplate, memory: ConversationBufferMemory, tools, db):
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
        self.db = db

    def __call__(self, state: AgentState) -> Dict:
        messages = state['messages']
        last_message = messages[-1]

        # Check if the last message is from Aurora (self-communication)
        if '__Aurora__' in last_message.content:
            # Remove the prefix and add a system message to indicate self-communication
            cleaned_content = last_message.content.replace("__Aurora__:", "").strip()
            user_message = messages[:-1] + [
                SystemMessage(content="You are now in a self-reflection mode. The following message is from yourself:"),
                AIMessage(content=cleaned_content)
            ]
        else:
            user_message = messages[-1].content if messages else ""



        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")

        files = self.db.query(File).filter(File.owner_id == state['user_id']).all()
        file_names = [file.filename for file in files]

        full_prompt = self.prompt.format(
            query=user_message,
            file_names=file_names,
        )

        ai_message = self.llm.invoke(full_prompt)

        # Store the message
        if state['sender'] == 'user':
            self.store_message(state['session_id'], user_message, "user")
        self.store_message(state['session_id'], ai_message.content, "AI")

        # Update memory
        self.memory.save_context({"input": user_message}, {"output": ai_message.content})

        return {
            "messages": messages + [AIMessage(content=ai_message.content)],
            "user_id": state['user_id'],
            "session_id": state['session_id'],
            "sender": "Aurora",
            "current_agent": "Aurora"
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

def router(state: AgentState) -> Literal["Aurora", "call_tool", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return END
    if "__Aurora__" in last_message.content:
        return "Aurora"
    return END

async def setup_langgraph_system(user, agent_config, session_id, db):
    llm = ChatGroq(
        temperature=0.1,
        groq_api_key=GROQ_API,
        model_name=agent_config['llm'],
        streaming=True
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    restore_memory_state(memory, session_id, db)

    llm_agent = create_agent(llm, agent_config['prompt'], memory, [fetch_customer_context, get_document_contents], db)

    llm_node = functools.partial(agent_node, agent=llm_agent, name="Aurora")

    workflow = StateGraph(AgentState)

    workflow.add_node("Aurora", llm_node)
    workflow.add_node("call_tool", tool_node)

    workflow.add_conditional_edges(
        "Aurora",
        router,
        {"call_tool": "call_tool", "__end__": END, "Aurora": "Aurora"},
    )

    workflow.add_edge(START, "Aurora")
    workflow.add_edge("call_tool", "Aurora")  # Tools always return to Aurora

    app = workflow.compile()

    return app

# Usage function
async def process_message(sys, message: str, user_id: str, session_id: int):
    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        user_id=user_id,
        session_id=session_id,
        sender="user",
        current_agent="Aurora"  # Start with Aurora as the default agent
    )

    async for chunk in sys.astream(initial_state):
        print(chunk)
        try:
            yield "__token__:" + chunk['Aurora']['messages'][-1].content
        except Exception:
            pass

def restore_memory_state(memory, session_id, db):
    messages = db.query(Message).filter_by(session_id=session_id).order_by(Message.timestamp)

    for message in messages:
        if message.sender == "user":
            memory.save_context({"input": message.content}, {"output": ""})
        else:
            memory.save_context({"input": ""}, {"output": message.content})

def create_agent(model, system_message:str, memory, tools, db):
    '''Creates an agent'''
    functions = [t for t in tools]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "\
            You are an AI assistant named Aurora. \
            Use the provided tools to progress towards answering the question. \
            If you are unable to answer, that's OK, prefix your answer with FINAL ANSWER to respond\
            to the user\n \
            For example: FINAL ANSWER: The answer to the meaning of life is 42 \n or \n \
            FINAL ANSWER: I don't know the answer to that \n\
            If you don't prefix your response you'll automatically reply to the user.\
            To consult with yourself, prefix your answer with __Aurora__, for example __Aurora__: This looks good enough\n\
            When you see a message starting with 'You are now in a self-reflection mode', it means you're \
            communicating with yourself. Treat this as an internal monologue or thought process.\n\
            You have access to the following tools and are encouraged to use them:{tool_names}. You can fetch a document's contents using the get_document_contents tool\n\
            Here's the data available in the user's knowledge base: {file_names}\n\
            {system_message}."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    prompt = prompt.partial(chat_history=memory.load_memory_variables({}).get("chat_history", ""))

    model.bind_tools(tools)
    llm = LLMNode(model, prompt, memory, tools, db)
    return llm

# Helper function to create a node for a given agent
def agent_node(state, agent, name) -> AgentState:
    result = agent.__call__(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        pass
    return {
        "messages": result['messages'],
        "user_id": state['user_id'],
        "session_id": state['session_id'],
        "sender": name,
        "current_agent": name
    }
