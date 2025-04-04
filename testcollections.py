import chromadb

client = chromadb.Client()
collections = client.list_collections()
print(collections)
