import pandas as pd
from transformers import pipeline

# --------------------------------------------------
# 1. CREATE INPUT DATASET
# --------------------------------------------------

data = {
    "id": [1, 2, 3],
    "text": [
        "I really enjoyed this product. It is excellent and very useful.",
        "The service was terrible and I am very disappointed.",
        "The product is okay. It works as expected."
    ]
}

df = pd.DataFrame(data)
df.to_csv("dataset/input_text.csv", index=False)


# --------------------------------------------------
# 2. SENTIMENT ANALYSIS
# --------------------------------------------------

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_results = []

for text in df["text"]:
    result = sentiment_pipeline(text)[0]

    sentiment_results.append({
        "text": text,
        "sentiment": result["label"],
        "score": result["score"]
    })

sentiment_df = pd.DataFrame(sentiment_results)
sentiment_df.to_csv(
    "dataset/sentiment_results.csv",
    index=False
)

print("Sentiment analysis completed.")


# --------------------------------------------------
# 3. TEXT SUMMARIZATION
# --------------------------------------------------

summary_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

long_text = """
Artificial intelligence is transforming many areas of modern life.
It is used in education, healthcare, banking, transportation and
business. AI systems can analyze large amounts of data and help
people make decisions. Machine learning allows computers to learn
patterns from data and improve their performance. Natural language
processing helps computers understand and generate human language.
As AI continues to develop, responsible and ethical use of technology
is important to ensure safety, privacy and fairness.
"""

summary_result = summary_pipeline(
    long_text,
    max_length=60,
    min_length=20,
    do_sample=False
)[0]["summary_text"]

summary_df = pd.DataFrame({
    "original_text": [long_text],
    "summary": [summary_result]
})

summary_df.to_csv(
    "dataset/summarized_text.csv",
    index=False
)

print("Text summarization completed.")


# --------------------------------------------------
# 4. TRANSLATION
# English → French
# --------------------------------------------------

translation_pipeline = pipeline(
    "translation_en_to_fr",
    model="Helsinki-NLP/opus-mt-en-fr"
)

translation_text = "Artificial intelligence is changing the world."

translation_result = translation_pipeline(
    translation_text
)[0]["translation_text"]

translation_df = pd.DataFrame({
    "english_text": [translation_text],
    "french_translation": [translation_result]
})

translation_df.to_csv(
    "dataset/translated_text.csv",
    index=False
)

print("Translation completed.")


# --------------------------------------------------
# 5. QUESTION ANSWERING
# --------------------------------------------------

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

context = """
Artificial intelligence is a branch of computer science that focuses
on creating machines and software that can perform tasks that normally
require human intelligence. AI is used in healthcare, education,
finance, transportation and many other fields.
"""

question = "What does artificial intelligence focus on?"

qa_result = qa_pipeline(
    question=question,
    context=context
)

qa_df = pd.DataFrame({
    "question": [question],
    "answer": [qa_result["answer"]],
    "score": [qa_result["score"]]
})

qa_df.to_csv(
    "dataset/qa_results.csv",
    index=False
)

print("Question answering completed.")


# --------------------------------------------------
# 6. PIPELINE COMPARISON
# --------------------------------------------------

comparison_data = {
    "Pipeline": [
        "Sentiment Analysis",
        "Text Summarization",
        "Translation",
        "Question Answering"
    ],
    "Purpose": [
        "Classifies text as positive or negative",
        "Creates a shorter version of a long text",
        "Translates text from English to French",
        "Answers questions using a given context"
    ]
}

comparison_df = pd.DataFrame(comparison_data)

comparison_df.to_csv(
    "dataset/pipeline_comparison.csv",
    index=False
)

print("Pipeline comparison completed.")

print("\nAll Hugging Face pipeline tasks completed successfully!")