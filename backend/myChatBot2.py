import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import chromadb
import arabic_reshaper
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


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the conversation interface, and handles the chat interaction.
    """
    # Get Groq API key
    groq_api_key = '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'
    model = 'meta-llama/llama-4-maverick-17b-128e-instruct'
    
    # Initialize Groq Langchain chat object
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )
    
    print("Hello! I'm your friendly assistant. I can help answer your questions, provide information, or just chat. Let's start our conversation!")

    system_prompt = 'You are a friendly assistant conversational chatbot, specialized in the food domain. You are going to interact entirely in Arabic. Along with the user input question, you will receive retrieved data from a database that is relevant to the user question. You should pass the retrieved data fully and as it is and do not use any quotation marks or any other symbols. Just check if the user question is related to the retrieved data and if it is not, just ignore the retrieved data and answer the user question.'

    conversational_memory_length = 5  # Number of previous messages the chatbot will remember

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    while True:
        user_question = input("Ask a question: ")

        # If the user has asked a question
        if user_question:
            # Enhance the query first to check if it is food-related
            query_enhancement_result = enhance_query_with_groq(user_question)
            print("Query Enhancement Result:", query_enhancement_result)

            if query_enhancement_result == "not food related" or query_enhancement_result == "respond based on chat history":
                # If the query is not food-related, skip retrieval
                retrieved_data = ""
                print("No food-related content found. Proceeding with generative response.")
            else:
                # If the query is food-related, proceed with retrieval
                suggested_data = choose_from_suggestions(query_enhancement_result)
                print("Selected Suggestion:", suggested_data)
                retrieved_data = retrieve_data(suggested_data)
                print("Retrieved Data:", retrieved_data)

            # Construct a chat prompt template using various components
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            # Fetch chat history as a list of messages
            chat_history = memory.load_memory_variables({})["chat_history"]

            # Combine the retrieved data and the user input for context
            full_prompt = prompt.format_messages(
                chat_history=chat_history,
                human_input=user_question
            )

            # Print the full prompt and the retrieved data being sent to the LLM
            print("Full Prompt Being Sent to the LLM:")
            print(full_prompt)
            print("Retrieved Data:")
            print(retrieved_data)

            # Add the retrieved data into the conversation context
            conversation_input = f"Retrieved Data: {retrieved_data}\nUser Question: {user_question}"

            # Create a conversation chain using the LangChain LLM
            conversation = LLMChain(
                llm=groq_chat,  # The Groq LangChain chat object initialized earlier
                prompt=prompt,  # The constructed prompt template
                verbose=False,   # Enable verbose output for debugging
                memory=memory,   # The conversational memory object
            )

            # The chatbot's answer is generated by sending the full prompt to the Groq API
            response = conversation.predict(human_input=conversation_input)
            print("Chatbot:", response)


def run_chatbot(user_question: str) -> str:
    """
    This function wraps the chatbot logic for use in an API or other interface.
    """
    groq_api_key = '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'
    model = 'meta-llama/llama-4-maverick-17b-128e-instruct'

    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    system_prompt = 'You are a friendly assistant conversational chatbot, specialized in the food domain. You are going to interact entirely in Arabic. Along with the user input question, you will receive retrieved data from a database that is relevant to the user question. You should pass the retrieved data fully and as it is and do not use any quotation marks or any other symbols. Just check if the user question is related to the retrieved data and if it is not, just ignore the retrieved data and answer the user question.'

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    query_enhancement_result = enhance_query_with_groq(user_question)

    if query_enhancement_result == "not food related" or query_enhancement_result == "respond based on chat history":
        retrieved_data = ""
    else:
        retrieved_data = retrieve_data(choose_from_suggestions(query_enhancement_result))

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    conversation_input = f"Retrieved Data: {retrieved_data}\nUser Question: {user_question}"

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    response = conversation.predict(human_input=conversation_input)
    return response


def run_chatbot_response_only(user_question: str, retrieved_data: str) -> str:
    """
    This function skips enhancement and suggestion logic. It uses provided retrieved_data directly.
    Used when a suggestion is selected from the frontend.
    """
    groq_api_key = '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'
    model = 'meta-llama/llama-4-maverick-17b-128e-instruct'

    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    system_prompt = 'You are a friendly assistant conversational chatbot, specialized in the food domain. You are going to interact entirely in Arabic. Along with the user input question, you will receive retrieved data from a database that is relevant to the user question. You should pass the retrieved data fully and as it is and do not use any quotation marks or any other symbols. Just check if the user question is related to the retrieved data and if it is not, just ignore the retrieved data and answer the user question.'

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    conversation_input = f"Retrieved Data: {retrieved_data}\nUser Question: {user_question}"

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    response = conversation.predict(human_input=conversation_input)
    return response


if __name__ == "__main__":
    main()
