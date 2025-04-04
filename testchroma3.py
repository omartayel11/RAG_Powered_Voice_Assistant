from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import arabic_reshaper

# Initialize model and ChromaDB client
model_name = "akhooli/Arabic-SBERT-100K"
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
client = chromadb.Client()

# Create or get the collection with the custom embedding function
collection = client.get_or_create_collection(
    name="recipes",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}  # Set the space for HNSW
)

recipes_data = [
    ("شوربة الحريرة", 
     "الحريرة شوربة مغربية رمضانية. والحريرة وإن كان أصلها مغربي، إلا أنها انتشرت في العالم العربي وأصبحت طبقًا مفضلًا، خاصة في شهر رمضان المبارك.",
     "كيلو مكعبات لحم غنم خالية من العظم والدهن، ملعقة صغيرة كركم، ملعقة صغيرة فلفل أسود مطحون، ملعقة صغيرة قرفة مطحونة، ملعقة صغيرة زنجبيل بودرة، ملعقة صغيرة فلفل أحمر حار مطحون، ملعقتان سمنة أو زبدة، كوبان كرفس، بصلة مفرومة، بصلة حمراء مفرومة، كوب كزبرة خضراء مفرومة، خمس حبات طماطم متوسطة الحجم، سبعة أكواب ماء، كوبان عدس أخضر، علبة حمص (٣٧٥ غرامًا) مصفاة من مائها، مئة غرام معكرونة عيدان رفيعة مثل الشعيرية، بيضتان، عصير ليمونة.",
     "في قدر كبير، توضع اللحمة مع الكركم والفلفل الأسود والقرفة والزنجبيل والفلفل الأحمر والسمنة والكرفس والبصل والكزبرة وتقلب معًا على نار هادئة بين الحين والآخر لمدة خمس دقائق. تقشر الطماطم وتقطع مكعبات كبيرة وتضاف للخليط وتترك القدر على نار هادئة لمدة خمس عشرة دقيقة. يضاف الماء والعدس للقدر وتعلو النار حتى يغلي الخليط، بعد الغليان توطى النار ويغطى القدر ويترك القدر على نار هادئة لمدة ساعتين. قبل انتهاء الشوربة بعشر دقائق ترفع الحرارة قليلاً لتصبح متوسطة، تضاف حبات الحمص والمعكرونة وتترك على النار لعشر دقائق حتى تستوي المعكرونة. يضاف عصير الليمون والبيض، تترك النار لدقيقتين ليستوي البيض ثم ترفع الشوربة وتقدم.",
     "شوربة", "recipe_1"),
    
    ("حساء البطاطا والسبانخ", 
     "برغم أن الكثير منا لم يتعود على استخدام السبانخ لغير طهي السبانخ أو في الفطائر، إلا أن هذه الشوربة طريقة لذيذة جدا لاستخدام السبانخ وهي أيضا مغذية جدا.",
     "ملعقتان زيت زيتون، بصلة متوسطة مقشرة ومفرومة، سن ثوم مبشور ناعم أو مدقوق، ملعقة صغيرة من أوراق الزعتر المجفف، أربعمائة غرام بطاطا مقشرة ومفرومة خشن، أربعة أكواب مرقة دجاج، حزمة سبانخ، نصف كوب مسحوق حليب، عصير نصف ليمونة، حبتي كراث كبير الحجم (اختياري).",
     "يسخن الزيت في قدر كبيرة ويضاف له البصل ثم الثوم والزعتر. تغطى القدر ويحمر الخليط على نار هادئة لمدة ثلاث إلى أربع دقائق مع التحريك من وقت لآخر. يغسل الكرات وترمى الأوراق الخارجية القاسية ويقطع قسمية الأبيض والأخضر إلى شرائح. يضاف الكرات مع البطاطا إلى خليط البصل. تغطى القدر وتترك على النار ثلاث إلى أربع دقائق أخرى مع التحريك مرة أو مرتين. يضاف مرق الدجاج وتترك القدر على النار حتى يغلي. تخفف النار وتترك القدر على نار هادئة لمدة عشرين دقيقة أو حتى تصبح البطاطا طرية. يضاف السبانخ وتترك القدر على النار ثلاث إلى أربع دقائق أخرى. يضرب خليط الحساء على دفعات في الخلاط الكهربائي. يضاف مسحوق الحليب للخليط. يقدم الحساء ساخنًا أو يترك حتى يبرد. قبل التقديم مباشرة يعصر قليل من عصير الليمون الحامض فوق كل وعاء تقديم.",
     "شوربة", "recipe_2"),
    
    ("شوربة البصل", 
     "شوربة البصل هي طبق شهي وسهل التحضير، يتم تحضيرها باستخدام البصل والزبدة والمرق، وتضاف لها التوابل لإعطاء نكهة مميزة.",
     "خمسة وعشرون جرام زبدة، ملعقتان طعام زيت زيتون، ستة بصيلات متوسطة الحجم، مقشرة ومقطعة شرائح رفيعة، ملعقة كبيرة سكر، أربعة أكواب مرقة لحم، ملح وفلفل أسود، القليل من الخل (حسب الرغبة).",
     "يسخن الزيت أو الزبدة في القدر. تضاف شرائح البصل وتحرك على نار خفيفة بين الوقت والآخر حتى يذبل البصل ويصبح لونه أصفر (من عشرين إلى خمسة وعشرين دقيقة). ينثر السكر فوق البصل مع استمرار التقليب فوق النار لمدة خمس دقائق. تضاف مرقة اللحم ثم يضاف الخل والملح والفلفل الأسود ويترك ليغلي لمدة خمس عشرة دقيقة. تقدم الشوربة مع شرائح الخبز الفرنسي المحمص. لعمل شرائح الخبز الفرنسي المحمص، يقطع رغيف خبز فرنسي إلى شرائح سميكة بعض الشيء وينثر فوقها جبن مبشور وتوضع تحت الشواية حتى يذوب الجبن.",
     "شوربة", "recipe_3")
]

for doc in recipes_data:
    # Insert the title
    collection.upsert(
        documents=[doc[0]],  # Title
        ids=[doc[5]],  # Shared ID
        metadatas=[{"type": "title", "category": doc[4]}]  # Metadata: "type" and "category"
    )

    # # Insert the description
    # collection.upsert(
    #     documents=[doc[1]],  # Description
    #     ids=[doc[5]],  # Shared ID
    #     metadatas=[{"type": "description", "category": doc[3]}]  # Metadata: "type" and "category"
    # )

    # # Insert the ingredients
    # collection.upsert(
    #     documents=[doc[2]],  # Ingredients
    #     ids=[doc[5]],  # Shared ID
    #     metadatas=[{"type": "ingredients", "category": doc[3]}]  # Metadata: "type" and "category"
    # )

    # Insert the instructions
    collection.upsert(
        documents=[doc[3]],  # Instructions
        ids=[doc[5]],  # Shared ID
        metadatas=[{"type": "instructions", "category": doc[4]}]  # Metadata: "type" and "category"
    )

def search_recipe(query_text):
    # Query by ingredients and instructions first (we're skipping title for simplicity here)
    results = collection.query(
        query_texts=[query_text], 
        n_results=4,
    )

    print("Search Results:")
    print(results)
    for result in results["documents"]:
        print(arabic_reshaper.reshape(result[0])[::-1])

search_recipe("يسطا بقولق ايهه بقا اسمع كده عايز طريقة شوربة البصل")  
