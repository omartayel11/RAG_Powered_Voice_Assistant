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
        # 👇 Step 1: Collect user info once at the start
        await websocket.send_json({
            "type": "collect_user_info",
            "message": "Let's get to know you! Please provide your name, gender (male/female), and profession (optional).",
            "fields": ["name", "gender", "profession"]
        })

        user_info = await websocket.receive_json()
        print("👤 Received user info:", user_info)

        name = user_info.get("name", "").strip()
        gender = user_info.get("gender", "").strip().lower()
        profession = user_info.get("profession", None)

        # Optional field handling
        profession = profession.strip() if profession else None

        # 👇 Step 2: Inject into chatbot
        session.set_user_info(name, gender, profession)

        # 👇 Step 3: Start main chat loop
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
