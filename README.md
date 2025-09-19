# Walmart with LLM Agent Interface

This project exposes Walmart weekly sales forecasts through:

1. **FastAPI** (`app/main.py`) – serves `/forecast` predictions from the trained per–store models in `models/artifacts`.
2. **LangGraph agent server** (`agent_server/graph.py`) – allows an LLM to translate natural-language chat into API calls so you can "talk" to the forecasting service.
3. **Streamlit UI** (`ui/streamlit_app.py`) – simple dashboard for manual queries.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt              # core forecasting app
pip install -r agent_server/requirements.txt # LangGraph + LLM helper
```

### 2. Run the FastAPI forecast service

```bash
uvicorn app.main:app --reload --port 8000
```

You can test it directly:

```bash
curl "http://127.0.0.1:8000/forecast?store_id=1&horizon=8&unit=weeks&model_name=auto"
```

### 3. Launch the LangGraph agent server

The agent wraps the `/forecast` endpoint with an LLM that extracts `store_id`, `horizon`, `unit`, and `model_name` from chat messages.

```bash
export WALMART_API="http://127.0.0.1:8000"  # FastAPI URL
export CHAT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # optional override
langgraph dev agent_server/graph.py
```

Then open the hosted or local Agent Chat UI and point it at the LangGraph server (see `agent_server/README.md` for details). Example prompts:

* "Forecast store 7 for 12 weeks with xgb."
* "Give me 3 months ahead for store 20 using prophet."

The agent validates ranges, caps horizons using the FastAPI limits, and returns the JSON response from the forecasting API for downstream rendering.

### 4. Optional: Streamlit dashboard

```bash
streamlit run ui/streamlit_app.py
```

## Repository layout

* `app/` – FastAPI application and forecasting helpers
* `train_*.py` – scripts for fitting Prophet / XGBoost models per store
* `agent_server/` – LangGraph server connecting an LLM to the forecast API
* `ui/` – Streamlit UI for manual exploration
* `data/` – (optional) raw data used during training

Feel free to swap in your own models inside `app/main.py` or point the agent to a remote API via `WALMART_API`.
