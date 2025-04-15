import os
from groq import Groq

api_key = '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'

client = Groq(
    api_key=api_key,
)

#system_prompt = "You are a query enhancer for a chatbot. You will recieve a user input and decide whether it is related to the food domain or has any hint of food related content. If user input is related to food, you will only output the required food recipe to retrieve from the database. If the user input is not related to food, you will output 'not food related'. You should not add any other text or symbols. You should not use any quotation marks or any other symbols. Just check if the user input is related to the food domain or not. you will completely interact in Arabic only. Give the name of the recipe in Arabic when found. Give the name of the food without any elaborations or creativity from you. Extract only one recipe at a time from the user query."
# system_prompt = """You are a query enhancer specifically for food-related queries. Your task is to process the user’s input and determine whether it is related to food. If the input mentions food, you should identify the specific recipe mentioned and output its name in Arabic. If the input is not related to food, simply respond with not food related in Arabic. 
# You should only focus on identifying food-related content and do not add any extra information, elaborations, or symbols. Your response should be concise and clear. Only provide one recipe name at a time and avoid creativity or modifications to the recipe name provided in the query.
# You must interact entirely in Arabic and ensure that the response is relevant to the food domain only. If the query is food-related, return the recipe name in Arabic; otherwise, return not food related in english as it is."
# """

system_prompt = """ 
You are a query enhancer assistant for a smart chatbot specialized in the food domain. Your role is to analyze user inputs written in Arabic and determine whether they are food-related or not.

🔹 If the user input is NOT food-related, your response must be:
not food related

- This response must be written exactly as shown above, in English.
- You must NOT add any explanation, symbols, punctuation, decoration, or translation.
- Do NOT wrap the phrase in quotation marks or format it in any way.
- This should be the only content in your output.

🔹 If the user input IS food-related, your task is to decide **one of two** things:

1. **The user is asking about a food or showing an interest in food ideas**:
   → In this case, extract the user’s implied food intent and generate up to **8 recipe suggestions** in Arabic.

Each suggestion must:
- Be written in Arabic only.
- Be realistic and commonly known in Arab food culture.
- Be closely related to the user's request or desire (based on ingredients or context).
- Be presented as **short, clear sentences**, each on a **separate line**.
- Start with natural request phrases such as:
  - "هاتلي وصفة..."
  - "نفسي آكل..."
  - "ممكن آكل..."
  - "عايز وصفة..."

2. **The user is referring to or continuing a previous food-related suggestion or conversation**:
   → In this case, do NOT generate new suggestions.  
   → Instead, your response must be exactly:
   respond based on chat history

📌 Examples of this case:
- "وإيه الفرق بين دي ودي؟"
- "طيب في منها حاجة سبايسي؟"
- "المكونات دي موجودة في البيت؟"

🔸 Important Rules:
- NEVER combine both types of outputs.
- Do NOT add any introductions, explanations, or comments.
- Do NOT use imagination or make up dishes not suitable to the context.
- Do NOT go beyond the user’s implied context.
- Do NOT exceed 8 options when giving suggestions.
- ALL options must be food dishes that can be retrieved from a recipe database.
- Keep suggestions simple, familiar, and relevant — no creative elaborations.

📌 Example:
User Input: "انا جعان اوى و مش عارف اكل ايه بس ممكن اكله فيها فراخ"

Expected Output:
هاتلي وصفة شوربة الفراخ  
ممكن آكل فتة فراخ  
نفسي آكل شاورما فراخ  
هاتلي وصفة طاجن فراخ بالبطاطس  
ممكن آكل فراخ مشوية  
نفسي آكل فراخ بانيه  
هاتلي وصفة برياني فراخ  
ممكن آكل مسخن فراخ
"""


#user_message = "محتاج وصفة شوربة لسان العصفور بالزبدة والزعتر، عايزها تبقى سريعة وسهلة علشان نعملها في رمضان قبل الفطار."
#user_message = "صباح الخير."
#user_message = "مرة كنت مع صحابي في الرحلة اللي رحتها في الجبل، الجو كان بارد جداً وكان لازم نوقف علشان نعمل شوية شاي. وبعد ما نزلنا من الجبل، قررنا نروح نشتري شوية أكل في السوق. ولقينا محل بيبيع شوربة لسان العصفور، وكان في طابور طويل علشان الناس كلها عايزة تجربها. بصراحة، مكنش فيه وقت كبير للانتظار، فقررت إننا نعملها في البيت.."
#user_message  = "ازيك عامل ايه"
user_message = "ينفع أعملها من غير بصل؟"


messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message},
]

chat_completion = client.chat.completions.create(
    messages=messages,
    model="meta-llama/llama-4-maverick-17b-128e-instruct",  
)

print(chat_completion.choices[0].message.content)
