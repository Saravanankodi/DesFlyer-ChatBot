from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag import ask_chatbot, stream_chatbot


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

@app.post("/chat/stream")
def chat_stream(data: Question):

    return StreamingResponse(
        stream_chatbot(data.question),
        media_type="text/plain"
    )
def voice_assistant():
    # 1. Detect user's voice
    # 2. Convert voice to text
    # 3. Send text to existing RAG chatbot
    # 4. Get streamed response
    # 5. Convert response to speech
    pass