import abc
from typing import Dict, List, Tuple, Annotated, TypedDict, Literal
from app.models.file import File
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
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
import json
import asyncio
from langsmith import traceable
import functools
from .config import mermaid_config

# Constants
GROQ_API = get_settings().GROQ_API
FILE_MATCH_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 0.15
TAG_MATCH_THRESHOLD = 0.8

@tool
def get_document_contents(document_name:Annotated[str, "Document name as seen in knowledge base"], user_id:Annotated[str, "user's id"]):
    """
    Retrieve/access document contents from knowledge base. Aurora can call this

    Args:
        document_name (str): Document name as seen in knowledge base.
        user_id (str): The ID of the user who owns the document.

    Returns:
        List[Dict]: A list of dictionaries, each containing a chunk's content and metadata. List is empty for empty files.
    """
    ingestion = Ingestion()

    results = ingestion.get_document_chunks(document_name, user_id)
    return results

# async def fetch_customer_context(query_str: Annotated[str, "User prompt"], user_id: Annotated[str, "user id"]) -> str:
#     """Fetch internal resources and user documents that might be relevant to the query.

#     Args:
#         query_str (str): Query string with relevant keywords from user's query.
#         user_id (str): The ID of the user.

#     Returns:
#         str: String containing information that might be relevant to the query.
#     """
#     ingestion = Ingestion()
#     ingestion.get_or_create_collection('embeddings')


#     # First, try to find file matches
#     # file_matches = ingestion.query_file_names(query_str, user_id)

#     # Then, try to find tag matches
#     tag_matches = await ingestion.query_by_tags(query_str, user_id)

#     # If we have strong file matches, prioritize those
#     # if len(file_matches) and file_matches[0]['similarity'] > FILE_MATCH_THRESHOLD:
#     #     return "\n".join([match['text'] for match in file_matches])

#     # If we have strong tag matches, include those
#     tag_matched_files = [match for match in tag_matches if match['combined_score'] > TAG_MATCH_THRESHOLD]

#     # Perform semantic search
#     semantic_results = await ingestion.query(query_str, user_id, relevance_threshold=RELEVANCE_THRESHOLD)

#     # Combine and rank results
#     combined_results = rank_and_combine_results(tag_matched_files, semantic_results)

#     # Extract and combine relevant text
#     context_text = []
#     for result in combined_results:
#         if 'text' in result:
#             context_text.append(result['text'])
#         elif 'sample_text' in result:
#             context_text.append(result['sample_text'])

#     return "\n".join(context_text)


tools = [get_document_contents]
tool_node = ToolNode(tools)
def rank_and_combine_results(tag_matches: List[Dict], semantic_results: List[Dict]) -> List[Dict]:
    combined_results = []



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
    def __init__(self, llm: ChatGroq, prompt: ChatPromptTemplate, memory: ConversationBufferWindowMemory, tools, db):
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
        self.db = db

    def __call__(self, state: AgentState) -> Dict:
        messages = state['messages']
        last_message = messages[-1]
        mode = 'user'

        print(type(last_message))
        # Check if the last message is from Aurora (self-communication)
        # if isinstance(last_message, ToolMessage):
        #     mode = 'tool'
        if state['sender'] == 'Aurora':
            # Remove the prefix and add a system message to indicate self-communication
            mode = 'self-reflection'


        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")

        files = self.db.query(File).filter(File.owner_id == state['user_id']).all()
        file_names = [file.filename for file in files]



        if isinstance(last_message, ToolMessage):
            full_prompt = self.prompt.format(
                query='',
                ai_query="here's the tool response: " + last_message.content,
                file_names=file_names,
                user_id=state['user_id'],
                mode=mode,
                chat_history=chat_history
            )
        elif state['sender'] == 'user':
            full_prompt = self.prompt.format(
                query=last_message.content,
                ai_query='',
                file_names=file_names,
                user_id=state['user_id'],
                mode=mode,
                chat_history=chat_history

            )
        else:
            full_prompt = self.prompt.format(
                query='',
                ai_query=last_message.content,
                file_names=file_names,
                user_id=state['user_id'],
                mode=mode,
                chat_history=chat_history

            )



        ai_message = self.llm.invoke(full_prompt)
        # Store the message
        if state['sender'] == 'user':
            if len(last_message.content):
                self.store_message(state['session_id'], last_message.content, "user")
            if len(ai_message.content):
                self.store_message(state['session_id'], ai_message.content, "AI", "__Aurora__" in ai_message.content)
        elif isinstance(last_message, ToolMessage):
            if len(last_message.content):
                self.store_message(state['session_id'], last_message.content, "Tool", True)
            if len(ai_message.content):
                self.store_message(state['session_id'], ai_message.content, "AI", "__Aurora__" in ai_message.content)
        elif state['sender'] == 'Aurora':
            if len(ai_message.content):
                self.store_message(state['session_id'], ai_message.content, "AI", "__Aurora__" in ai_message.content)


        # Update memory
        self.memory.save_context({"input": last_message.content}, {"output": ai_message.content})

        return {
            "messages": messages + [ai_message],
            "user_id": state['user_id'],
            "session_id": state['session_id'],
            "sender": "Aurora",
            "current_agent": "Aurora"
        }

    def store_message(self, session_id: int, content: str, sender: str, hidden:bool = False):
        new_message = Message(session_id=session_id, content=content, sender=sender, hidden=hidden)
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
    elif "__Aurora__" in last_message.content:
        return "Aurora"
    return END

async def setup_langgraph_system(user, agent_config, session_id, db):
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API,
        model_name=agent_config['llm'],
        streaming=True
    ).bind_tools(tools)

    memory = ConversationBufferWindowMemory(memory_key="chat_history",k=10, return_messages=True)
    restore_memory_state(memory, session_id, db)

    llm_agent = create_agent(llm, agent_config['prompt'], memory, [get_document_contents], db)

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
        try:
            payload =  chunk['Aurora']['messages'][-1].content
            if '__Aurora__' not in payload:
                yield "__token__: " + payload
        except Exception:
            pass

def restore_memory_state(memory, session_id, db):
    messages = db.query(Message).filter_by(session_id=session_id).order_by(Message.timestamp)

    for message in messages:
        if message.sender == "user":
            memory.chat_memory.add_user_message(message.content)
        elif message.sender == "AI":
            memory.chat_memory.add_ai_message(message.content)
        elif message.sender == "Tool":
            memory.chat_memory.add_ai_message(message.content)

def create_agent(model, system_message:str, memory, tools, db):
    '''Creates an agent'''
    prompt = ChatPromptTemplate.from_messages([
        ("system", "\
            You have these modes: 'user', 'self-reflection' and 'tool'\nIn 'user' mode, you're receiving input from the user,\
            in 'self-reflection' you're thinking to yourself, and finally in 'tool' mode you've just received information from a tool call.\n\
            You are currently in {mode} mode\n\
            You are an AI assistant named Aurora. You've been developed by Arctic Labs. \
            If you are unable to answer, that's OK. You don't have to have context to answer, do the best you can with what you know.\n \
            To consult with yourself, prefix your answer with __Aurora__, for example __Aurora__: This looks good enough.\n\
            Make sure to consult with yourself before any tool call to plan out how and why you're using the tool.\
            If you don't prefix with your response with '__Aurora__', you'll automatically reply to the user.\
            Whenever prompted make sure that you eventually reply to the user!\
            When in self-reflection mode (Using __Aurora__ flag), it means you're \
            communicating with yourself. Treat this as an internal monologue or thought process. This is where you can plan how you're going to\
             complete the user's query and or plan which tools you're going to call and with what parameters. If you're missing any information (don't hallucinate the details in this mode) from then user you can always\
             escape this state/mode omitting the prefix and just with the response to the user. If you're unsure about some function/tool parameters don't make them up in your monologue!\n\
            You are encouraged to plan out your thoughts first!\n\
            Do not use flags __Aurora__ in your response with the user.\
            You have access to the following tools [{tool_names}].  \
            You can use this tool to access these uploaded files present in the users knowledge base, here are the file names: {file_names}.\n\
            If a tool returns falsely or empty content, do not call it again! Move on and try to answer without context but tell the user what you're doing.\n\
            I'll reiterate, If you've just received received a tool do not call that tool again!\n \
            Similarly if you just received a duplicate message from yourself don't respond to yourself again... The user is waiting\
            for a response!\n\
            You SHOULD NOT explain unnecessary information like 'I need to access the file first', it's redundent. Do not announce tool usage to the user, rather to yourself in your internal monologue ex. __Aurora__: I need to fetch the 'document.pdf' file in order to summarise the notes.\
            Do not use the tool to fetch irrelevant files! This is a waste of time and resources, be mindful of the user's query.\
            Try your best to answer even if there's no file in the knowledge base that seems relevant!\
            You have been trained on a massive corpus of data, and are well equiped to answer the users query, you do not need the knowledge base to answer the users questions.\
            Here's the user's user_id:{user_id}. You don't and should not ask the user for it. \n\
            Your response to the user should be formated in ReactMarkdown, the following plugins are used: remarkGfm, remarkMath, rehypeKatex, rehypeStringify. You good writing techniques like headings, subheadings and bold and italics were applicable.\
            There's support for GitHub-specific extensions (remarkGfm): tables, strikethrough, tasklists, and literal URLs. For example | Feature | Support | | ---------: | :------------------- | | CommonMark | 100% | | GFM | 100% w/ remark-gfm | \n\
            When providing links use appropriate format like ex.[<ins>link to example</ins>](https://example.com). Make sure to underline links.\n\
            Further more you use mermaid to draw diagrams to illustrate ideas and concepts the user:\n\
            {mermaid}\
            {system_message}.\n\
            Chat History: {chat_history}\n\
            ----------------------"),
        ("human", "{query}"),
        ("ai", "{ai_query}"),
    ])
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(mermaid=mermaid_config)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    # prompt = prompt.partial(chat_history=memory.load_memory_variables({}).get("chat_history", ""))

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
