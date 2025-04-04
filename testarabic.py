import arabic_reshaper

# text_to_be_reshaped =  'اللغة العربية رائعة'

# reshaped_text = arabic_reshaper.reshape(text_to_be_reshaped)

# rev_text = reshaped_text[::-1]  # slice backwards 

# print(rev_text)


# -*- coding: utf-8 -*-
# s = "ذهب الطالب الى المدرسة"
# with open("file.txt", "w", encoding="utf-8") as myfile:
#     myfile.write(s)


# file = open("firstChunkTrial.txt", encoding="utf-8")
# text_to_be_reshaped = file.read()
# reshaped_text = arabic_reshaper.reshape(text_to_be_reshaped)
# rev_text = reshaped_text[::-1]  # slice backwards
# print(rev_text)


file = open("firstChunkTrial.txt","r", encoding="utf-8")
print(file.read())