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
from fastapi import Request, HTTPException
from utils import create_user, get_user_by_email, hash_password, verify_password, add_recipe_to_favourites
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

@app.post("/signup")
async def signup(request: Request):
    data = await request.json()

    # Check required fields only
    required_fields = ["email", "password", "gender"]
    for field in required_fields:
        if not data.get(field):
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    # Optional fields with defaults
    optional_fields = {
        "name": "",
        "profession": "",
        "allergies": [],
        "likes": [],
        "dislikes": [],
        "favorite_recipes": []
    }

    for key, default in optional_fields.items():
        data[key] = data.get(key, default)

    existing_user = await get_user_by_email(data["email"])
    if existing_user:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = await create_user(data)
    return {"message": "User created successfully", "user_id": user_id}

@app.post("/login")
async def login(request: Request):
    data = await request.json()

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    user = await get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    return {
        "message": "Login successful",
        "email": user["email"]
    }

@app.post("/add-favourite")
async def add_favourite(request: Request):
    data = await request.json()
    email = data.get("email")
    title = data.get("title")
    recipe = data.get("recipe")

    if not all([email, title, recipe]):
        raise HTTPException(status_code=400, detail="Missing data.")

    result = await add_recipe_to_favourites(email, title, recipe)
    return result

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket connection established.")

    session = WebSocketBotSession()

    try:
        # Step 1: Wait for email (identifier)
        await websocket.send_json({
            "type": "auth_request",
            "message": "من فضلك ادخل البريد الإلكتروني لتسجيل الدخول."
        })

        login_info = await websocket.receive_json()
        user_email = login_info.get("email", "").strip()

        user_data = await get_user_by_email(user_email)
        if not user_data:
            await websocket.send_json({
                "type": "error",
                "message": "المستخدم غير موجود. من فضلك سجل أولاً."
            })
            await websocket.close()
            return

        # Step 2: Use user data from DB to set session
        session.set_user_info(
            name=user_data.get("name", ""),
            gender=user_data.get("gender", "male"),
            profession=user_data.get("profession", None),
            likes = user_data.get("likes", []),
            dislikes = user_data.get("dislikes", []),
            allergies = user_data.get("allergies", []),
            favorite_recipes = user_data.get("favorite_recipes", []),
        )

        session.user_email = user_email  # (optional for future reference)

        # Step 3: Start the chat loop
        while True:
            user_message = await websocket.receive_text()
            print(f"\n📨 Incoming WebSocket message: {user_message}")

            # Check for reset command
            if user_message.strip() == "/new":
                session = WebSocketBotSession()  # Reset session completely
                session.set_user_info(
                    name=user_data.get("name", ""),
                    gender=user_data.get("gender", "male"),
                    profession=user_data.get("profession", None),
                    likes = user_data.get("likes", []),
                    dislikes = user_data.get("dislikes", []),
                    allergies = user_data.get("allergies", []),
                    favorite_recipes = user_data.get("favorite_recipes", [])
                )
                session.user_email = user_email

                await websocket.send_json({
                    "type": "reset",
                    "message": "✅ تم بدء محادثة جديدة تمامًا."
                })

                continue

            if session.expecting_choice:
                try:
                    selected_index = int(user_message.strip()) - 1
                    result = await session.handle_choice(selected_index)
                except (ValueError, IndexError):
                    result = {
                        "type": "error",
                        "message": "من فضلك اختر رقم من الاختيارات الموجودة."
                    }
            else:
                result = await session.handle_message(user_message)


            await websocket.send_json(result)
            print("📤 Response sent to frontend.\n")

    except WebSocketDisconnect:
        print("🔴 WebSocket disconnected.")
