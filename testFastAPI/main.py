from fastapi import FastAPI
from schemas import HealthResponse, AskRequest, AskResponse
from services import generate_answer

app = FastAPI()


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "alive"}


@app.post("/ask", response_model=AskResponse)
async def ask(data: AskRequest):
    result = generate_answer(data.question)
    return result
