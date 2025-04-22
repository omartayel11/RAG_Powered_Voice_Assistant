import chromadb

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_collection(name="recipestest")

all_docs = collection.get()
all_ids = all_docs["ids"]

if all_ids:
    collection.delete(ids=all_ids)
    print(f" Deleted {len(all_ids)} documents from the collection.")
else:
    print(" Collection is already empty.")
