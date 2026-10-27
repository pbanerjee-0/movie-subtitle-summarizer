'''Initial TF-IDF Summarizer

Unigram and bigram based extractive text summarizer to generate movie and episode summaries based 
on preprocessed subtitle text files.

'''

import pandas as pd
import spacy as nlp
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_colwidth', None)
