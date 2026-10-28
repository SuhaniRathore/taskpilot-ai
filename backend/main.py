from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from backend.config import APP_NAME
from backend.database import init_db, save_task, get_recent_tasks
from backend.services.agent import run_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="A tool-using personal AI agent with document RAG and data analysis.",
    lifespan=lifespan,
)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: list[dict] = Field(default_factory=list)

@app.get("/health")
def health():
    return {"status": "ok", "app_name": APP_NAME}

@app.post("/chat")
def chat(request: ChatRequest):
    result = run_agent(request.message, request.history)
    save_task(request.message, result["answer"], result.get("tools_used", []))
    return result

@app.get("/tasks")
def tasks(limit: int = 30):
    return {"tasks": get_recent_tasks(limit)}
