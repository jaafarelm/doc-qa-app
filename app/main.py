from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "alive"}


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    question = payload.question
    return {"answer": f"You asked: {question}"}