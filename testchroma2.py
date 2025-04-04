from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

model_name = "akhooli/Arabic-SBERT-100K"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="my_recipes", 
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}  # Set the space for HNSW
)



documents = [
    "الحريرة شوربة مغربية رمضانية",
    "شوربة العدس مع الليمون"
]

collection.upsert(
    documents=documents,
    ids=["id1", "id2"],
)

results = collection.query(
    query_texts=["شوربة مغربية"], # Chroma will embed this for you
    n_results=1 # how many results to return
)

print(results)
