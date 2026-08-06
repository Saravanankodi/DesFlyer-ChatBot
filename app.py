from fastapi import FastAPI
from pydantic import BaseModel

from rag import ask_chatbot


app = FastAPI(
    title="DesFlyer Chatbot API",
    description="RAG based chatbot using Gemma 2B",
    version="1.0"
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "DesFlyer Chatbot API is running"
    }


@app.post("/chat")
def chat(data: Question):

    answer = ask_chatbot(data.question)

    return {
        "question": data.question,
        "answer": answer
    }