# import chromadb
# from sentence_transformers import SentenceTransformer
# chroma_client = chromadb.Client()

# # switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
# collection = chroma_client.get_or_create_collection(name="recipes")

# model = SentenceTransformer("akhooli/Arabic-SBERT-100K")

# # Define a sample recipe (حريرة)
# recipe_data = {
#     "title": "الحريرة",  # Recipe title
#     "category": "soup",  # Recipe category
#     "ingredients": [
#         "كيلو مكعبات لحم غنم خالية من العظم والدهن",
#         "1 ملعقة صغيرة كركم",
#         "1 ملعقة صغيرة فلفل أسود مطحون",
#         "1 ملعقة صغيرة قرفة مطحونة",
#         "1/4 ملعقة صغيرة زنجبيل بودرة",
#         "1/4 ملعقة صغيرة فلفل أحمر حار مطحون",
#         "2 ملعقة سمنة أو زبدة",
#         "3/4 كوب كرفس",
#         "1 بصلة مفرومة",
#         "1 بصلة حمراء مفرومة",
#         "1/2 كوب كزبرة خضراء مفرومة",
#         "5 حبات طماطم متوسطة الحجم",
#         "7 كوب ماء",
#         "3/4 كوب عدس أخضر",
#         "375 غرام حمص (مصفاة من مائها)",
#         "100 غرام معكرونة عيدان رفيعة مثل الشعيرية",
#         "2 بيض",
#         "عصير ليمونة"
#     ],
#     "instructions": """
#     في قدر كبير، توضع اللحمة مع الكركم والفلفل الأسود والقرفة والزنجبيل والفلفل الأحمر والسمنة والكرفس.
#     تقلب معًا على نار هادئة بين الحين والآخر لمدة 5 دقائق. تقشر الطماطم وتقطع مكعبات كبيرة وتضاف للخليط
#     وتترك القدر على نار هادئة لمدة 15 دقيقة. يضاف الماء والعدس للقدر وتُعلي النار حتى يغلي الخليط، بعد الغليان
#     تُوطي النار ويغطى القدر ويترك على نار هادئة لمدة ساعتين.
#     قبل انتهاء الشوربة بعشر دقائق ترفع الحرارة قليلاً لتصبح متوسطة، وتضاف حبات الحمص والمعكرونة وتترك على
#     النار لعشر دقائق حتى تستوي المعكرونة. يُضاف عصير الليمون والبيض، وتترك النار لدقيقتين ليُستوى البيض
#     ثم ترفع الشوربة وتقدم.
#     """,
#     "embedding": None  # We'll add the embedding after we generate it
# }

# recipe_embedding = model.encode([recipe_data["instructions"]])[0]
# recipe_data["embedding"] = recipe_embedding


# # switch `add` to `upsert` to avoid adding the same documents every time
# collection.upsert(
#     documents=[recipe_data["instructions"]],
#     ids=[recipe_data["title"]],  # Use the title as ID (unique identifier)
#     metadatas=[{
#         "recipe_type": recipe_data["category"],
#         "title": recipe_data["title"],
#         "ingredients": ", ".join(recipe_data["ingredients"]),
#     }],
#     embeddings=[recipe_data["embedding"]]
# )

# # Example user query
# query_text = "دجاج مشوي"

# # Generate the embedding for the query
# query_embedding = model.encode([query_text])[0]

# # Query the ChromaDB collection to get the most similar recipes
# results = collection.query(
#     query_embeddings=[query_embedding],
#     n_results=1  # Get the top 5 most similar recipes
# )

# # Print out the results
# for i, result in enumerate(results['documents']):
#     print(f"Result {i+1}:")
#     print(f"Document: {result}")
#     print(f"Metadata: {results['metadatas'][i]}")
#     print(f"Distance: {results['distances'][i]}")
#     print("\n" + "-"*40 + "\n")



import chromadb
chroma_client = chromadb.Client()

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "شربة طماطم",
        "اجنحة الدجاج المشوية",
        "مكرونة بصلصة الطماطم"
    ],
    metadatas=[{"category":"soup"},{"category":"appetizer"},{"category":"pasta"}],
    ids=["id1", "id2", "id3"]
)

results = collection.query(
    query_texts=["هاتلى وصفة شربة الطماطم"], # Chroma will embed this for you
    n_results=1 # how many results to return
)

print(results)

