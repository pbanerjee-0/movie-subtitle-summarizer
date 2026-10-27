'''Initial TF-IDF Summarizer

Unigram and bigram based extractive text summarizer to generate movie and episode summaries based 
on preprocessed subtitle text files.

'''

import pandas as pd
import spacy
import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_colwidth', None)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"]) #  reduce processing time by disabling

def spaCy_text_tokenize(text):
    tokens=[]
    for token in nlp(text):
        if not token.is_punct and not token.is_space and not token.is_stop:
            tokens.append(token.lemma_.lower())

    return tokens

# Start with a test subtitle text file
with open('data/subtitles/agatha_all_along_s01e05.txt', 'r', encoding='utf-8') as f:
    subtitle_lines = [line.strip() for line in f if line.strip()]

vectorizer = TfidfVectorizer(tokenizer=spaCy_text_tokenize, token_pattern=None, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(subtitle_lines)

print(tfidf_matrix.shape)

# print(vectorizer.get_feature_names_out()) 

sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

# Pair each sentence with its original index and score
scored_sentences = list(enumerate(sentence_scores))

# Sort by score in descending order and pick the top N sentences (e.g., top 3)
num_summary_sentences = 8
top_ranked = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:num_summary_sentences]

# Sort them back by their original index (chronological order)
top_ranked_chronological = sorted(top_ranked, key=lambda x: x[0])

summary_sentences = [subtitle_lines[idx] for idx, score in top_ranked_chronological]
generated_summary = " ".join(summary_sentences)

print("\n--- Generated Summary ---")
print(generated_summary)