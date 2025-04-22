import os
import re
from docx import Document

# --- Config ---
input_docx_path = "Copy of recipes.docx"  # Change this to your actual file
output_folder = "cleaned_recipes_arabic"
os.makedirs(output_folder, exist_ok=True)

# --- Symbol Cleanup ---
bullet_symbols = r"[•\-–—►▪▪️•✦●○✓✔❖→➤➔⬤*❌✅🔸🔹⬛⬜🔴🟠🟡🟢🔵🟣🟤⚫⚪🔺🔻⬆️⬇️⬅️➡️❗‼️…]+"
english_numbering = r"^\s*\d+[\.\)]\s*"  # Matches: 1. or 1)

# --- Number & Fraction Conversion Maps ---
arabic_digits = {
    "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤",
    "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩"
}

fraction_map = {
    "1/2": "نصف",
    "½": "نصف",
    "1/4": "ربع",
    "¼": "ربع",
    "1/5": "خمس",
    "⅕": "خمس",
    "3/4": "ثلاثة أرباع",
    "¾": "ثلاثة أرباع",
    "1/3": "ثلث",
    "⅓": "ثلث",
    "2/3": "ثلثين",
    "⅔": "ثلثين",
    "1/8": "ثمن",
    "⅛": "ثمن",
    "3/8": "ثلاثة أثمان",
    "⅜": "ثلاثة أثمان",
    "5/8": "خمسة أثمان",
    "⅝": "خمسة أثمان",
    "7/8": "سبعة أثمان",
    "⅞": "سبعة أثمان",
    
}

# --- Helper functions ---
def convert_english_digits_to_arabic(text):
    return ''.join(arabic_digits.get(char, char) for char in text)

def replace_fractions(text):
    for eng, arabic in fraction_map.items():
        text = text.replace(eng, arabic)
    return text

def clean_text(text):
    text = re.sub(bullet_symbols, "", text)
    text = re.sub(english_numbering, "", text)
    text = replace_fractions(text)
    text = convert_english_digits_to_arabic(text)
    text = re.sub(r"[^\w\s\u0600-\u06FF.,!?؛،]", "", text)  # Keep Arabic and punctuation
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip()

def extract_cleaned_recipes(doc_path):
    document = Document(doc_path)
    recipes = []
    current_recipe = []

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            if current_recipe:
                recipes.append("\n".join(current_recipe))
                current_recipe = []
            continue
        cleaned = clean_text(text)
        if cleaned:
            current_recipe.append(cleaned)

    if current_recipe:
        recipes.append("\n".join(current_recipe))

    return recipes

def save_recipes_to_files(recipes, output_path):
    for i, recipe in enumerate(recipes, 1):
        file_path = os.path.join(output_path, f"recipe_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(recipe)

# --- Run the script ---
if __name__ == "__main__":
    recipes = extract_cleaned_recipes(input_docx_path)
    save_recipes_to_files(recipes, output_folder)
    print(f"✅ {len(recipes)} recipes cleaned, converted to Arabic numerals, and saved to '{output_folder}'")
