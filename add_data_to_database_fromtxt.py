import os
import chromadb
from chromadb.utils import embedding_functions

recipes_folder = "cleaned_recipes_arabic"
collection_name = "recipestest"
model_name = "akhooli/Arabic-SBERT-100K"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
chroma_client = chroma_client = chromadb.HttpClient(host='localhost', port=8000)

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}
)

for filename in os.listdir(recipes_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(recipes_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                continue

            title = lines[0].strip()  # First line is the title
            full_recipe = "".join(lines).strip()

        # Insert into ChromaDB
        collection.upsert(
            documents=[full_recipe],
            ids=[title],  # Use title as the ID
            metadatas=[{
                "type": "recipe",
                "title": title
            }]
        )

print(f"✅ All recipes from '{recipes_folder}' have been inserted into ChromaDB with titles as IDs.")
