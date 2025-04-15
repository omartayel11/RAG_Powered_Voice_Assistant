import os
from groq import Groq

api_key = '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'

client = Groq(
    api_key=api_key,
)

system_prompt = "You are a query enhancer for a chatbot. You will recieve a user input and decide whether it is related to the food domain or has any hint of food related content. If user input is related to food, you will only output the required food recipe to retrieve from the database. If the user input is not related to food, you will output 'not food related'. You should not add any other text or symbols. You should not use any quotation marks or any other symbols. Just check if the user input is related to the food domain or not. you will completely interact in Arabic only. Give the name of the recipe in Arabic when found. Give the name of the food without any elaborations or creativity from you. Extract only one recipe at a time from the user query."
# system_prompt = """You are a query enhancer specifically for food-related queries. Your task is to process the user’s input and determine whether it is related to food. If the input mentions food, you should identify the specific recipe mentioned and output its name in Arabic. If the input is not related to food, simply respond with not food related in Arabic. 
# You should only focus on identifying food-related content and do not add any extra information, elaborations, or symbols. Your response should be concise and clear. Only provide one recipe name at a time and avoid creativity or modifications to the recipe name provided in the query.
# You must interact entirely in Arabic and ensure that the response is relevant to the food domain only. If the query is food-related, return the recipe name in Arabic; otherwise, return not food related in english as it is."
# """

#user_message = "محتاج وصفة شوربة لسان العصفور بالزبدة والزعتر، عايزها تبقى سريعة وسهلة علشان نعملها في رمضان قبل الفطار."
#user_message = "صباح الخير."
user_message = "مرة كنت مع صحابي في الرحلة اللي رحتها في الجبل، الجو كان بارد جداً وكان لازم نوقف علشان نعمل شوية شاي. وبعد ما نزلنا من الجبل، قررنا نروح نشتري شوية أكل في السوق. ولقينا محل بيبيع شوربة لسان العصفور، وكان في طابور طويل علشان الناس كلها عايزة تجربها. بصراحة، مكنش فيه وقت كبير للانتظار، فقررت إننا نعملها في البيت.."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message},
]

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama3-70b-8192",  
)

print(chat_completion.choices[0].message.content)
