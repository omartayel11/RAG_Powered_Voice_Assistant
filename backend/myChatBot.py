from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq


def retrieve_data(query):
    """
    This function retrieves relevant data from the ChromaDB database based on the user's query.
    """
    # Initialize Chroma client
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)

    # Load the custom embedding function (ensure you have initialized it)
    model_name = "akhooli/Arabic-SBERT-100K"
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    try:
        collection = chroma_client.get_collection("recipestest", embedding_function=sentence_transformer_ef)
        print("Collection 'recipestest' found.")
    except chromadb.errors.InvalidCollectionException:
        print("Collection 'recipestest' does not exist. Please add data first.")
        return ""

    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=1,
    )

    # Return the relevant retrieved documents without reshaping
    if results['documents']:
        response = "\n".join([result[0] for result in results["documents"]])
        return response
    else:
        return "Sorry, I couldn't find any relevant information in my database."


def enhance_query_with_groq(query):
    """
    This function uses the Groq API to enhance the query and determine if it's food-related.
    """
    # Manually insert the API key
    api_key = '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'

    # Initialize the Groq client with the manual API key
    client = Groq(api_key=api_key)

    system_prompt = """ 
You are a query enhancer assistant for a smart chatbot specialized in the food domain. Your role is to analyze user inputs written in Arabic and determine whether they are food-related or not.

🔹 If the user input is NOT food-related, your response must be:
not food related

- This response must be written exactly as shown above, in English.
- You must NOT add any explanation, symbols, punctuation, decoration, or translation.
- Do NOT wrap the phrase in quotation marks or format it in any way.
- This should be the only content in your output.

🔹 If the user input IS food-related, your task is to decide **one of two** things:

1. **The user is asking about a food or showing an interest in food ideas or recipes**:
   → In this case, extract the user’s implied food intent and generate a small list of **recipe suggestions** in Arabic.

Each suggestion must:
- Be written in Arabic only.
- Be realistic and commonly known in Arab food culture.
- Be directly relevant to the user's request or desire (based on ingredients or context).
- Be presented as **short, clear sentences**, each on a **separate line**.
- Start with natural request phrases such as:
  - "هاتلي وصفة..."
  - "نفسي آكل..."
  - "ممكن آكل..."
  - "عايز وصفة..."

 **Important rules for suggestion generation**:
- **Only generate suggestions if the user is clearly asking for food ideas or a recipe**.
  - If the user just says they're hungry or vague (e.g., "أنا جعان"), do NOT give suggestions.
  - Wait for another input that specifies a type of food.
- **If the user mentions a very specific dish or food (e.g., "كشري" or "مقلوبة فراخ")**, it's okay to suggest only **one recipe** closely matching that request.
- **Default behavior is to keep suggestions minimal**, ideally between **1–3**.
- Only increase the number of suggestions (up to a max of 8) **if** the user is vague or mentions broad food categories (e.g., "عايز أكلة فيها فراخ").

2. **The user is referring to or continuing a previous food-related suggestion or conversation**:
   → In this case, do NOT generate new suggestions.  
   → Instead, your response must be exactly:
   respond based on chat history

Examples of this case:
- "وإيه الفرق بين دي ودي؟"
- "طيب في منها حاجة سبايسي؟"
- "المكونات دي موجودة في البيت؟"

 Additional Important Rules:
- NEVER mix both types of output.
- Do NOT add explanations, commentary, or introduction.
- Do NOT invent unrealistic dishes.
- Do NOT exceed 8 suggestions.
- Do NOT repeat similar suggestions using different phrasing.
- ALL suggestions must be recipes that exist and are likely available in the recipe database.

Example:
User Input: "انا جعان اوى و مش عارف اكل ايه بس ممكن اكله فيها فراخ"

Expected Output:
هاتلي وصفة شوربة الفراخ  
ممكن آكل فتة فراخ  
نفسي آكل شاورما فراخ  
هاتلي وصفة طاجن فراخ بالبطاطس  
ممكن آكل فراخ مشوية  
نفسي آكل فراخ بانيه  
هاتلي وصفة برياني فراخ  
ممكن آكل مسخن فراخ

 Summary of Suggestion Logic:
- If vague hunger: → respond based on chat history
- If specific dish: → 1 suggestion is enough
- If general request with a food type: → 1–3 suggestions
- If broad or open-ended: → up to 8 suggestions max
- Always prefer fewer suggestions when possible

You are only responsible for generating suggestions if — and only if — the user is clearly asking for food ideas or recipes.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="meta-llama/llama-4-maverick-17b-128e-instruct",  # Change to appropriate model for query enhancement
    )

    return chat_completion.choices[0].message.content


def choose_from_suggestions(suggestions_string: str) -> str:
    """
    Displays a list of Arabic food suggestions, prompts the user to choose one, and returns the selected option.
    """
    suggestions = [line.strip() for line in suggestions_string.strip().split('\n') if line.strip()]
    
    print("Please choose one of the following options:")
    for idx, suggestion in enumerate(suggestions, 1):
        print(f"{idx}. {suggestion}")
    
    selected_index = None
    while selected_index is None:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(suggestions):
                selected_index = choice - 1
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    return suggestions[selected_index]

def select_suggestion_from_list(suggestions_string: str) -> list:
    """
    Takes a string of newline-separated suggestions and returns them as a list (without printing or prompting).
    """
    return [line.strip() for line in suggestions_string.strip().split('\n') if line.strip()]


class WebSocketBotSession:
    def __init__(self):
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        self.expecting_choice = False
        self.suggestions = []
        self.original_question = ""
        self.groq_api_key = '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'
        self.model = 'meta-llama/llama-4-maverick-17b-128e-instruct'
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name=self.model)
        self.system_prompt = """
You're a smart, friendly chatbot with a light sense of humor, and you're all about food. You speak entirely in Arabic, specifically in the Egyptian dialect. You’re not a boring assistant — you're more like a foodie friend helping the user figure out what to eat.

 Your job:
- Talk to the user naturally and casually — like a close friend, not a robot.
- Try to guide the user toward mentioning a type of food or a specific recipe they’re craving.
- If the user hints at a food item, a **query enhancer** kicks in and generates food suggestions.
- If the user selects one of the suggestions, the system retrieves a recipe from the database.
- Once you receive the recipe, **you must display it exactly as it is** — no translation, no formatting, no quotation marks, no emojis. Just the raw text.

 Be aware of the chat history:
- Sometimes, the user will ask a follow-up question about a previous recipe (e.g., “is it spicy?”, “how do I cook it?”).
- You should remember the last few interactions and use them to keep the conversation flowing naturally.

 If the user is just chatting out of boredom:
- It’s okay to go a little off-topic at first.
- Joke around, be funny, ask light questions — but gently **steer the conversation back to food** when possible. That’s your comfort zone.

 Behavior Guidelines:
- Never act formal or robotic. No “as an AI model...” replies. You’re a foodie with personality.
- If the user is unclear (e.g., “I’m hungry”), ask follow-up questions like: “Craving meat? Chicken? Sweet stuff?”
- If the user goes too far from the food domain, steer them back playfully.
- If a recipe is retrieved, do not change it in any way — just deliver it plainly.
- Keep suggestions, questions, and answers short, natural, and full of flavor — just like a good meal.

 System Flow (for your awareness):
1. User sends a message.
2. If food is mentioned, the **query enhancer** suggests dishes.
3. User selects a dish.
4. The system retrieves the recipe.
5. You show the recipe exactly as it is, and continue the conversation.

 Your ultimate goal: Make the user feel like they’re chatting with a foodie friend who always knows what’s good to eat.

Be smart, be warm, and always bring it back to food.
"""

    async def handle_message(self, user_input: str):
        print(f"\n🟡 Received user message: {user_input}")
        self.original_question = user_input

        query_result = enhance_query_with_groq(user_input)
        print(f"🟢 Query Enhancement Result:\n{query_result}\n")

        if query_result in ["not food related", "respond based on chat history"]:
            retrieved_data = ""
            print("🔍 No food-related content. Proceeding with empty retrieval.\n")
            return await self._generate_response(user_input, retrieved_data)

        self.suggestions = select_suggestion_from_list(query_result)
        self.expecting_choice = True

        print("📋 Suggestions extracted:")
        for i, suggestion in enumerate(self.suggestions, 1):
            print(f"{i}. {suggestion}")
        print()

        return {
            "type": "suggestions",
            "message": "اختر رقم من الاختيارات التالية:",
            "suggestions": self.suggestions
        }

    async def handle_choice(self, choice_index: int):
        print(f"🟠 User selected choice index: {choice_index}")
        if 0 <= choice_index < len(self.suggestions):
            selected = self.suggestions[choice_index]
            print(f"✅ Selected Suggestion: {selected}")
            retrieved_data = retrieve_data(selected)
            print(f"📦 Retrieved Data:\n{retrieved_data}\n")
            self.expecting_choice = False
            return await self._generate_response(self.original_question, retrieved_data)
        else:
            print("❌ Invalid choice index received.")
            return {
                "type": "error",
                "message": "اختيار غير صالح. حاول رقم تاني."
            }

    async def _generate_response(self, user_input: str, retrieved_data: str):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ])

        chat_history = self.memory.load_memory_variables({})["chat_history"]
        print(f"📚 Chat History Size: {len(chat_history)}")

        full_prompt = prompt.format_messages(
                chat_history=chat_history,
                human_input=user_input
            )

            # Print the full prompt and the retrieved data being sent to the LLM
        print("Full Prompt Being Sent to the LLM:")
        print(full_prompt)

        conversation_input = f"Retrieved Data: {retrieved_data}\nUser Question: {user_input}"
        print("🧠 Sending to LLM:\n", conversation_input)

        conversation = LLMChain(
            llm=self.groq_chat,
            prompt=prompt,
            verbose=False,
            memory=self.memory,
        )

        response = conversation.predict(human_input=conversation_input)

        print("💬 Chatbot Response:\n", response)
        return {
            "type": "response",
            "message": response
        }

