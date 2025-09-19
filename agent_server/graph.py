# agent_server/graph.py
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama   # or any Chat* model you like
import requests, os

BASE_URL = os.getenv("FORECAST_API", "http://127.0.0.1:8000")

class State(TypedDict):
    messages: List[BaseMessage]

@tool
def walmart_forecast(query: str) -> str:
    """
    Call the Walmart forecast API. Input example:
    'forecast store 1 for 8 weeks (auto)' or
    'forecast store 12 for 5 months with prophet'
    """
    # super-light parser (you already have a better one in your repo)
    import re
    m_store = re.search(r"store\s+(\d+)", query, re.I)
    m_h    = re.search(r"(\d+)\s*(weeks|months?)", query, re.I)
    m_model = re.search(r"(prophet|auto)", query, re.I)
    if not (m_store and m_h):
        return "Please specify like: 'forecast store 7 for 12 weeks (auto)'."

    store_id = int(m_store.group(1))
    horizon  = int(m_h.group(1))
    unit     = "months" if "month" in m_h.group(2).lower() else "weeks"
    model    = m_model.group(1) if m_model else "auto"

    # Your API is GET-based:
    params = dict(store_id=store_id, horizon=horizon, unit=unit, model_name=model)
    try:
        r = requests.get(f"{BASE_URL}/forecast", params=params, timeout=30)
        r.raise_for_status()
        return r.text  # UI will show this; you can JSON pretty-print if you prefer
    except requests.RequestException as e:
        return f"API error: {e}"

def build_graph():
    llm = ChatOllama(model="llama3", temperature=0)  # or ChatOpenAI(...)
    llm = llm.bind_tools([walmart_forecast])

    def call_model(state: State):
        # Standard “LLM with tools” pattern:
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    g = StateGraph(State)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    g.add_edge("assistant", END)
    return g.compile()

# Expose for the LangGraph server to discover
graph = build_graph()
