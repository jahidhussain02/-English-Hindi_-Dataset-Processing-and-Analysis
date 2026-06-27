import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Rename columns
df.columns = ["English", "Hindi"]

# Remove missing values
df = df.dropna()

# Word count
df["English_Word_Count"] = df["English"].str.split().str.len()
df["Hindi_Word_Count"] = df["Hindi"].str.split().str.len()

# Keep between 5 and 50 words
df = df[
    (df["English_Word_Count"] >= 5) &
    (df["English_Word_Count"] <= 50) &
    (df["Hindi_Word_Count"] >= 5) &
    (df["Hindi_Word_Count"] <= 50)
]

# Difference
df["Difference"] = (
    df["English_Word_Count"] -
    df["Hindi_Word_Count"]
)

# Keep only -10 to +10
df = df[
    (df["Difference"] >= -10) &
    (df["Difference"] <= 10)
]

# Save
df.to_excel("cleaned_dataset.xlsx", index=False)

print(df.head())
print(len(df))
import pandas as pd
from transformers import pipeline

# Load cleaned dataset
df = pd.read_excel("cleaned_dataset.xlsx")

# Select first 100 sentences
df = df.head(100)

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-hi"
)

translations = []

for sentence in df["English"]:
    result = translator(sentence)
    translations.append(result[0]["translation_text"])

df["Generated_Hindi"] = translations

df.to_excel("translated.xlsx", index=False)
from sacrebleu.metrics import BLEU, CHRF, TER
import pandas as pd

df = pd.read_excel("translated.xlsx")

references = [[x] for x in df["Hindi"]]
hypotheses = df["Generated_Hindi"].tolist()

bleu = BLEU()
chrf = CHRF()
ter = TER()

bleu_score = bleu.corpus_score(
    hypotheses,
    list(zip(*references))
)

chrf_score = chrf.corpus_score(
    hypotheses,
    list(zip(*references))
)

ter_score = ter.corpus_score(
    hypotheses,
    list(zip(*references))
)

with open("scores.txt","w",encoding="utf-8") as f:
    f.write(f"BLEU : {bleu_score.score}\n")
    f.write(f"CHRF : {chrf_score.score}\n")
    f.write(f"TER : {ter_score.score}\n")

print(bleu_score.score)
print(chrf_score.score)
print(ter_score.score)