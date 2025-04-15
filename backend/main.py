from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from myChatBot import run_chatbot

app = FastAPI()

# Allow requests from your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_question = request.message
    response = run_chatbot(user_question)
    return {"response": response}
