from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download from the 🤗 Hub
model = SentenceTransformer("akhooli/Arabic-SBERT-100K")
# Run inference
# sentences = [
#     'ما هو نوع الدهون الموجودة في الأفوكادو',
#     'حوالي 15 في المائة من الدهون في الأفوكادو مشبعة ، مع كل كوب واحد من الأفوكادو المفروم يحتوي على 3.2 جرام من الدهون المشبعة ، وهو ما يمثل 16 في المائة من DV البالغ 20 جرامًا. تحتوي الأفوكادو في الغالب على دهون أحادية غير مشبعة ، مع 67 في المائة من إجمالي الدهون ، أو 14.7 جرامًا لكل كوب مفروم ، ويتكون من هذا النوع من الدهون.',
#     'يمكن أن يؤدي ارتفاع مستوى الدهون الثلاثية ، وهي نوع من الدهون (الدهون) في الدم ، إلى زيادة خطر الإصابة بأمراض القلب ، ويمكن أن يؤدي توفير مستوى مرتفع من الدهون الثلاثية ، وهي نوع من الدهون (الدهون) في الدم ، إلى زيادة خطر الإصابة بأمراض القلب. مرض.',
# ]

sentences = ['ادينيى الوصفه بتاعة شربة الطماطم','انا بحب الطماطم اوى و عايز اعمل شربة الطماطم']
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]

similarities = cosine_similarity(embeddings)

# Print the similarity matrix
print("Similarity Matrix:")
print(similarities)

# Output the similarity scores between each pair of sentences
print("\nSimilarity Scores:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"Similarity between Sentence {i + 1} and Sentence {j + 1}: {similarities[i][j]:.4f}")