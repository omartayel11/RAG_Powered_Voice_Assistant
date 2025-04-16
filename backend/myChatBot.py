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

1. **The user is asking about a food or showing an interest in food ideas**:
   → In this case, extract the user’s implied food intent and generate up to **8 recipe suggestions** in Arabic.

Each suggestion must:
- Be written in Arabic only.
- Be realistic and commonly known in Arab food culture.
- Be closely related to the user's request or desire (based on ingredients or context).
- Be presented as **short, clear sentences**, each on a **separate line**.
- Start with natural request phrases such as:
  - "هاتلي وصفة..."
  - "نفسي آكل..."
  - "ممكن آكل..."
  - "عايز وصفة..."

2. **The user is referring to or continuing a previous food-related suggestion or conversation**:
   → In this case, do NOT generate new suggestions.  
   → Instead, your response must be exactly:
   respond based on chat history

 Examples of this case:
- "وإيه الفرق بين دي ودي؟"
- "طيب في منها حاجة سبايسي؟"
- "المكونات دي موجودة في البيت؟"

🔸 Important Rules:
- NEVER combine both types of outputs.
- Do NOT add any introductions, explanations, or comments.
- Do NOT use imagination or make up dishes not suitable to the context.
- Do NOT go beyond the user’s implied context.
- Do NOT exceed 8 options when giving suggestions.
- ALL options must be food dishes that can be retrieved from a recipe database.
- Keep suggestions simple, familiar, and relevant — no creative elaborations.

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
        self.system_prompt = 'You are a friendly assistant conversational chatbot, specialized in the food domain. You are going to interact entirely in Arabic. Along with the user input question, you will receive retrieved data from a database that is relevant to the user question. You should pass the retrieved data fully and as it is and do not use any quotation marks or any other symbols. Just check if the user question is related to the retrieved data and if it is not, just ignore the retrieved data and answer the user question.'

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

