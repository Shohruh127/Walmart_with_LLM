
# Walmart Forecast — LangGraph Agent Server + Agent Chat UI

This folder provides a minimal LangGraph server that lets an LLM call your existing FastAPI `/forecast` endpoint.
You can use **LangChain Agent Chat UI** as a frontend.

## Prereqs
- Your FastAPI service running locally (e.g., `uvicorn app.main:app --reload --port 8000`)
- Python 3.10+
- Node 18+ (for Agent Chat UI)
- If using Ollama for local LLM: `ollama serve` and `ollama pull llama3`

## 1) Install server deps
```bash
pip install -r requirements.txt
```

## 2) Run the LangGraph dev server
```bash
export FORECAST_API="http://127.0.0.1:8000"   # your FastAPI URL
# If using OpenAI, also set: export OPENAI_API_KEY=sk-...
langgraph dev agent_server/graph.py
```

You should see a local server (default port printed by the CLI). It implements the standard LangGraph API with a `messages` state.

## 3) Launch Agent Chat UI and point it to your LangGraph server

**Option A: Use the hosted UI**
- Go to https://agentchat.vercel.app
- Deployment URL: your LangGraph server URL (e.g., `http://localhost:2024`)
- Assistant/Graph ID: `agent` (default)

**Option B: Run locally**
```bash
npx create-agent-chat-app
cd agent-chat-ui
pnpm install   # or npm/yarn
# Create .env from the example below
echo "NEXT_PUBLIC_API_URL=http://localhost:2024" > .env.local
echo "NEXT_PUBLIC_ASSISTANT_ID=agent" >> .env.local
pnpm dev
```

Then open http://localhost:3000 and chat with your assistant. Example messages:
- `forecast store 1 for 8 weeks (auto)`
- `forecast store 12 for 5 months with prophet`

## Notes
- If you see 405 on `/forecast` with POST, this server uses GET with query params to match your FastAPI.
- For CORS: since the UI talks to the LangGraph server (not directly to FastAPI), CORS on FastAPI is not required.
- For production: host LangGraph and set `NEXT_PUBLIC_API_URL` to that URL in your UI.
