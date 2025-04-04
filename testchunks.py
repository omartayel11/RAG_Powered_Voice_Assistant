# import arabic_reshaper

# def chunk_text(text):
#     # Split the text into lines and identify sections
#     lines = text.split('\n')
#     chunks = []
#     chunk = ""
    
#     for line in lines:
#         if "المقادير" in line or "الطريقة" in line:
#             if chunk:
#                 chunks.append(chunk)
#             chunk = line  # Start a new chunk for this section
#         else:
#             chunk += "\n" + line  # Append to the current chunk
    
#     if chunk:
#         chunks.append(chunk)  # Add the final chunk
    
#     # Reshape and reverse the text for correct Arabic rendering
#     reshaped_chunks = [arabic_reshaper.reshape(chunk)[::-1] for chunk in chunks]
    
#     return reshaped_chunks

# # Example usage:
# file = open("firstChunkTrial.txt", encoding="utf-8")
# text = file.read()
# chunks = chunk_text(text)

# # Now, print the chunks
# for i, chunk in enumerate(chunks, 1):
#     print(f"Chunk {i}:")
#     print(chunk)
#     print("\n" + "-"*40 + "\n")



from transformers import AutoTokenizer, AutoModel
import torch
import arabic_reshaper

# Load AraBERT model and tokenizer
model_name = "aubmindlab/bert-base-arabertv2"  # AraBERTv2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def get_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get the model's output (last hidden state)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The embeddings are in the last hidden state, we can average across all tokens (or use the CLS token representation)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Averaging over token embeddings
    
    return embeddings

# Function to chunk the text into sections and generate embeddings
def chunk_text(text):
    # Split the text into lines and identify sections
    lines = text.split('\n')
    chunks = []
    chunk = ""
    
    for line in lines:
        if "المقادير" in line or "الطريقة" in line:
            if chunk:
                chunks.append(chunk)
            chunk = line  # Start a new chunk for this section
        else:
            chunk += "\n" + line  # Append to the current chunk
    
    if chunk:
        chunks.append(chunk)  # Add the final chunk
    
    # Reshape and reverse the text for correct Arabic rendering
    reshaped_chunks = reshaped_chunks = [arabic_reshaper.reshape(chunk)[::-1] for chunk in chunks]
    
    # Generate embeddings for each reshaped chunk
    embeddings = [get_embeddings(chunk) for chunk in reshaped_chunks]
    
    return embeddings, reshaped_chunks

# Example usage:
file = open("firstChunkTrial.txt", encoding="utf-8")
text = file.read()
embeddings, chunks = chunk_text(text)

# Now, print the embeddings and chunks
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
    print(f"Chunk {i}:")
    print(f"Text: {chunk}")
    print(f"Embedding: {embedding}\n")
    print("-" * 40)
