# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from myChatBot import run_chatbot

# app = FastAPI()

# # Allow requests from your React app
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, replace with actual domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     message: str

# @app.post("/chat")
# def chat_endpoint(request: ChatRequest):
#     user_question = request.message
#     response = run_chatbot(user_question)
#     return {"response": response}



from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from myChatBot import WebSocketBotSession

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket connection established.")
    session = WebSocketBotSession()

    try:
        while True:
            user_message = await websocket.receive_text()
            print(f"\n📨 Incoming WebSocket message: {user_message}")

            if session.expecting_choice:
                try:
                    selected_index = int(user_message.strip())
                    result = await session.handle_choice(selected_index)
                except ValueError:
                    print("❌ Received non-numeric input while expecting a choice.")
                    result = {
                        "type": "error",
                        "message": "من فضلك اختر رقم صحيح من القائمة."
                    }
            else:
                result = await session.handle_message(user_message)

            await websocket.send_json(result)
            print("📤 Sent response to frontend.\n")

    except WebSocketDisconnect:
        print("🔴 WebSocket disconnected.")

