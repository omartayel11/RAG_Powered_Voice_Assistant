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
    allow_origins=["*"],  # In production, set your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}  # Optional: session management for multi-user apps

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket connection established.")
    
    session = WebSocketBotSession()

    try:
        # Step 1: Ask for user info once at the beginning
        await websocket.send_json({
            "type": "collect_user_info",
            "message": "أهلاً! قبل ما نبدأ، من فضلك اديني اسمك، النوع (ذكر أو أنثى)، والمهنة (اختياري).",
            "fields": ["name", "gender", "profession"]
        })

        user_info = await websocket.receive_json()
        print("👤 Received user info:", user_info)

        name = user_info.get("name", "").strip()
        gender = user_info.get("gender", "").strip().lower()
        profession = user_info.get("profession", "").strip() or None

        session.set_user_info(name, gender, profession)

        # Step 2: Main chat loop
        while True:
            user_message = await websocket.receive_text()
            print(f"\n📨 Incoming WebSocket message: {user_message}")

            if session.expecting_choice:
                # Expecting a number for recipe choice
                try:
                    selected_index = int(user_message.strip()) - 1  # User sees 1-based index
                    result = await session.handle_choice(selected_index)
                except (ValueError, IndexError):
                    print("❌ Invalid choice input.")
                    result = {
                        "type": "error",
                        "message": "من فضلك اختر رقم من الاختيارات الموجودة."
                    }
            else:
                # Normal user input
                result = await session.handle_message(user_message)

            await websocket.send_json(result)
            print("📤 Response sent to frontend.\n")

    except WebSocketDisconnect:
        print("🔴 WebSocket disconnected.")
