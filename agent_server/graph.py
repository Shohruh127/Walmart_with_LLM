# agent_server/graph.py
from typing import List, TypedDict, Dict, Any, Annotated
import os, re, json, requests

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ===== Config =====
# Point this to your running Walmart FastAPI service
BASE_URL = os.getenv("WALMART_API", "http://127.0.0.1:8000")  # must expose GET /forecast
MAX_WEEKS = int(os.getenv("MAX_WEEKS", "20"))
SUPPORTED_MODELS = {"auto", "xgb", "prophet"}
_SUPPORTED_MODELS_STR = "|".join(f'"{m}"' for m in sorted(SUPPORTED_MODELS))

# ===== Lightweight local LLM via Hugging Face =====
# (Works on CPU; uses GPU automatically if available)
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = os.getenv("CHAT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",  # GPU if present, otherwise CPU
)
_hf = pipeline(
    "text-generation",
    model=_mdl,
    tokenizer=_tok,
    max_new_tokens=128,
    do_sample=False,  # deterministic → stable JSON
)
LLM = HuggingFacePipeline(pipeline=_hf)

# ===== Graph state =====
class AgentState(TypedDict):
    # Agent Chat UI expects a `messages` key in state
    messages: Annotated[List[BaseMessage], add_messages]

# ===== Helpers =====
def _llm_extract(user_text: str) -> Dict[str, Any]:
    """Ask the LLM to emit STRICT JSON with the 4 keys."""
    sys = (
        "Return ONLY JSON with keys: store_id (int), horizon (int), "
        f'unit ("weeks"|"months"), model_name ({_SUPPORTED_MODELS_STR}).'
    )
    prompt = f"{sys}\nUser: {user_text}\nJSON:"
    out = LLM.invoke(prompt).content
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _regex_fallback(text: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    m_store = re.search(r"store\s*(\d+)", text, re.I)
    m_h = re.search(r"(\d+)\s*(weeks?|months?)", text, re.I)
    m_model = re.search(r"(prophet|auto|xgb)", text, re.I)
    if m_store:
        d["store_id"] = int(m_store.group(1))
    if m_h:
        d["horizon"] = int(m_h.group(1))
        d["unit"] = "months" if "month" in m_h.group(2).lower() else "weeks"
    d["model_name"] = m_model.group(1).lower() if m_model else "auto"
    return d

def _validate_and_fix(p: Dict[str, Any]):
    notes = []
    s = int(p.get("store_id") or 1)
    h = int(p.get("horizon") or 8)
    u = str(p.get("unit") or "weeks").lower().strip()
    m = str(p.get("model_name") or "auto").lower().strip()

    if not (1 <= s <= 45):
        notes.append(f"store_id {s} out of range → clamped 1..45")
        s = max(1, min(45, s))
    if u not in ("weeks", "months"):
        notes.append("unit not recognized → weeks")
        u = "weeks"
    if m not in SUPPORTED_MODELS:
        notes.append("model not recognized → auto")
        m = "auto"

    # Cap horizon respecting MAX_WEEKS regardless of unit requested
    weeks_requested = h if u == "weeks" else h * 4
    if weeks_requested > MAX_WEEKS:
        notes.append(f"horizon capped to {MAX_WEEKS} weeks")
        if u == "weeks":
            h = MAX_WEEKS
        else:
            # Convert back to months rounding down so we never exceed MAX_WEEKS
            h = max(1, MAX_WEEKS // 4)
    if h <= 0:
        notes.append("horizon <= 0 → set to 1")
        h = 1

    return {"store_id": s, "horizon": h, "unit": u, "model_name": m}, notes

def _call_forecast(params: Dict[str, Any]):
    try:
        r = requests.get(f"{BASE_URL}/forecast", params=params, timeout=60)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return r.text
    except Exception as e:
        return {"ok": False, "message": f"API error: {e}", "params": params}

# ===== Node =====
def respond(state: AgentState) -> AgentState:
    # get last user message
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    user_text = last_user.content if last_user else ""

    parsed = _llm_extract(user_text) or _regex_fallback(user_text)
    params, notes = _validate_and_fix(parsed)
    result = _call_forecast(params)

    payload = {"request": user_text, "params_used": params, "notes": notes, "result": result}
    reply = AIMessage(content="```json\n" + json.dumps(payload, indent=2) + "\n```")
    return {"messages": [reply]}

# ===== Graph =====
builder = StateGraph(AgentState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile()
