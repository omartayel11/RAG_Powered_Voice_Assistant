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

    system_prompt = 'You are a friendly assistant conversational chatbot, specialized in the food domain. You are going to interact intirely in Arabic. along with the user input question, you will recieve retrieved data from a database that is relevant to the user question. You should pass the retrieved data fully and as it is and do not use any quotation marks or any other symbols. just check if the user  question is related to the retrieved data and if it is not, just ignore the retrieved data and answer the user question.'


    conversational_memory_length = 5  # Number of previous messages the chatbot will remember

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    while True:
        user_question = input("Ask a question: ")

        # If the user has asked a question
        if user_question:
            # Retrieve relevant data from ChromaDB based on the user's query
            retrieved_data = retrieve_data(user_question)

            # Construct a chat prompt template using various components
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=system_prompt
                    ),  # The persistent system prompt that is always included at the start of the chat

                    MessagesPlaceholder(
                        variable_name="chat_history"
                    ),  # This placeholder will be replaced by actual chat history

                    HumanMessagePromptTemplate.from_template(
                        "{human_input}"
                    ),  # This template is where the user's current input will be injected into the prompt
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
                memory=memory,  # The conversational memory object
            )

            # The chatbot's answer is generated by sending the full prompt to the Groq API
            response = conversation.predict(human_input=conversation_input)
            print("Chatbot:", response)

if __name__ == "__main__":
    main()
