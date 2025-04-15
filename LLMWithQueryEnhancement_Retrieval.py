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

    system_prompt = "You are a query enhancer for a chatbot. You will receive a user input and decide whether it is related to the food domain or has any hint of food-related content. If user input is related to food, you will only output the required food recipe to retrieve from the database. If the user input is not related to food, you will output 'not food related'. You should not add any other text or symbols. You should not use any quotation marks or any other symbols. Just check if the user input is related to the food domain or not. you will completely interact in Arabic only. Give the name of the recipe in Arabic when found. Give the name of the food without any elaborations or creativity from you. Extract only one recipe at a time from the user query."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",  # Change to appropriate model for query enhancement
    )

    return chat_completion.choices[0].message.content


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the conversation interface, and handles the chat interaction.
    """
    # Get Groq API key
    groq_api_key = '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'
    model = 'llama3-70b-8192'
    
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

            if query_enhancement_result == "not food related":
                # If the query is not food-related, skip retrieval
                retrieved_data = ""
                print("No food-related content found. Proceeding with generative response.")
            else:
                # If the query is food-related, proceed with retrieval
                retrieved_data = retrieve_data(query_enhancement_result)

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


if __name__ == "__main__":
    main()
