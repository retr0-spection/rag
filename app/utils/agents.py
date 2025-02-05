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
from app.database import get_db, get_settings
from app.models.contexthistory import ContextHistory
from app.models.models import Agent, Session as ChatSession, Message
import re
import json
import asyncio
from langsmith import traceable
import functools
from .config import mermaid_config, values, code_formatting
from app.utils.tools import web_search_and_extract, get_document_contents, run_python_code_in_container
from app.utils.prompt import aurora_prompt, aurora_prompt_lite
# Constants
GROQ_API = get_settings().GROQ_API
FILE_MATCH_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 0.2
TAG_MATCH_THRESHOLD = 0.8

tools = [get_document_contents, web_search_and_extract, run_python_code_in_container]
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
    current_agent: str
    tool_calls: int
    error: bool  # New field to track error state
    error_message: str  # New field to store error messages

class LLMNode:
    def __init__(self, llm: ChatGroq, prompt: ChatPromptTemplate, memory: ConversationBufferWindowMemory, tools, db):
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
        self.db = db

    def __call__(self, state: AgentState) -> Dict:
        try:
            messages = state['messages']
            last_message = messages[-1]
            # Reset tool calls counter when processing a user message
            if isinstance(last_message, HumanMessage):
                state["tool_calls"] = 0

            chat_history = self.memory.load_memory_variables({}).get("chat_history", "")

            files = self.db.query(File).filter(File.owner_id == state['user_id']).all()
            file_names = [file.filename for file in files]

            if isinstance(last_message, ToolMessage):
                forced_prompt = f"""
                A tool has returned the following response: {last_message.content}

                Please process this response before considering any new tool calls.
                If the information is sufficient, respond to the user.
                If you absolutely need another tool call, explain why.
                """
                full_prompt = self.prompt.format(
                    query='',
                    ai_query=forced_prompt,
                    file_names=file_names,
                    user_id=state['user_id'],
                    chat_history=chat_history
                )
            elif state['sender'] == 'user':
                full_prompt = self.prompt.format(
                    query=last_message.content,
                    ai_query='',
                    file_names=file_names,
                    user_id=state['user_id'],
                    chat_history=chat_history
                )
            else:
                full_prompt = self.prompt.format(
                    query='',
                    ai_query=last_message.content,
                    file_names=file_names,
                    user_id=state['user_id'],
                    chat_history=chat_history
                )

            try:
                ai_message = self.llm.invoke(full_prompt)
            except Exception as e:
                print(e)
                if "rate limit" in str(e).lower():
                    return {
                        "error": True,
                        "error_message": "Rate limit exceeded. Please try again in a moment.",
                        "messages": messages,
                        "user_id": state['user_id'],
                        "session_id": state['session_id'],
                        "sender": "Aurora",
                        "current_agent": "Aurora",
                        "tool_calls": state['tool_calls']
                    }
                else:
                    raise e

            # Store the message
            if state['sender'] == 'user':
                if len(last_message.content):
                    self.store_message(state['session_id'], last_message.content, "user")
                if len(ai_message.content):
                    self.store_message(state['session_id'], ai_message.content, "AI", "__Aurora__" in ai_message.content and "__exit__" not in ai_message.content)
            elif isinstance(last_message, ToolMessage):
                if len(last_message.content):
                    self.store_message(state['session_id'], last_message.content, "Tool", True)
                if len(ai_message.content):
                    self.store_message(state['session_id'], ai_message.content, "AI", "__Aurora__" in ai_message.content and "__exit__" not in ai_message.content)
            elif state['sender'] == 'Aurora':
                if len(ai_message.content):
                    self.store_message(state['session_id'], ai_message.content, "AI", "__Aurora__" in ai_message.content and "__exit__" not in ai_message.content)

            # Update memory
            self.memory.save_context({"input": last_message.content}, {"output": ai_message.content})

            return {
                "error": False,
                "messages": messages[-4:] + [ai_message], #not more than 5 previous messages
                "user_id": state['user_id'],
                "session_id": state['session_id'],
                "sender": "Aurora",
                "current_agent": "Aurora",
                "tool_calls": state['tool_calls']
            }
        except Exception as e:
            return {
                "error": True,
                "error_message": f"An unexpected error occurred: {str(e)}",
                "messages": messages,
                "user_id": state['user_id'],
                "session_id": state['session_id'],
                "sender": "Aurora",
                "current_agent": "Aurora",
                "tool_calls": state['tool_calls']
            }

    def store_message(self, session_id: int, content: str, sender: str, hidden:bool = False):
        new_message = Message(session_id=session_id, content=content, sender=sender, hidden=hidden)
        self.db.add(new_message)
        self.db.commit()


def router(state: AgentState) -> Literal["Aurora", "call_tool", "__end__"]:
    # First check for errors
    if state.get("error", False):
        return END  # Exit if there's an error

    messages = state["messages"]
    last_message = messages[-1]

    # If we've made too many tool calls, force an end
    if state["tool_calls"] >= 2:
        _m = last_message
        _m.content = "We're stuck in a loop. Stop making tool calls and respond to the user's query appropriately!"
        messages[-1] = _m
        return "Aurora"

    # Check if this is a tool response being processed
    if isinstance(last_message, ToolMessage):
        return "Aurora"  # Always process tool responses

    # For AI messages, check for tool calls or exit
    if isinstance(last_message, AIMessage):
        if "__exit__" in last_message.content:
            return END
        if last_message.tool_calls:
            state["tool_calls"] += 1  # Increment tool call counter
            return "call_tool"
        if "__Aurora__" in last_message.content:
            return "Aurora"

    return END

async def setup_langgraph_system(user, agent_config, session_id, db):
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API,
        model_name=agent_config['llm'],
        streaming=True
    ).bind_tools(tools)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    restore_memory_state(memory, session_id, db)

    llm_agent = create_agent(llm, agent_config['prompt'], memory, tools, db)

    llm_node = functools.partial(agent_node, agent=llm_agent, name="Aurora")

    # Setup workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("Aurora", llm_node)
    workflow.add_node("call_tool", tool_node)

    workflow.add_conditional_edges(
        "Aurora",
        router,
        {
            "call_tool": "call_tool",
            "__end__": END,
            "Aurora": "Aurora"
        }
    )

    workflow.add_edge(START, "Aurora")
    workflow.add_edge("call_tool", "Aurora")

    app = workflow.compile()

    return app

# Usage function
async def process_message(sys, message: str, user_id: str, session_id: int):
    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        user_id=user_id,
        session_id=session_id,
        sender="user",
        current_agent="Aurora",
        tool_calls=0,
        error=False,
        error_message=""
    )

    async for chunk in sys.astream(initial_state):
        try:
            if 'Aurora' in chunk:
                if chunk['Aurora'].get('error', False):
                    yield f"__error__: {chunk['Aurora']['error_message']}"
                    return

                payload = chunk['Aurora']['messages'][-1].content
                if '__exit__' in payload:
                    yield "__token__: " + payload.split('__exit__')[1]
                elif '__Aurora__' not in payload:
                    yield "__token__: " + payload
        except Exception as e:
            yield f"__error__: An unexpected error occurred: {str(e)}"
            return


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
        ("system", aurora_prompt),
        ("human", "{query}"),
        ("ai", "{ai_query}"),
    ])
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(mermaid=mermaid_config)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    prompt = prompt.partial(values=values)
    prompt = prompt.partial(code_formatting=code_formatting)
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
