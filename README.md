# Marvel Subtitle Summarizer

This script is a TF-IDF extractive summary tool designed to generate summaries for Marvel movies and TV shows from subtitle text files, built on unigrams and bigrams. 

This is the traditional (n-gram) baseline for a group NLP project. Subtitle download scripts are located in the group repo: https://github.com/TanishqBansod/marvelmovie

## Task

- **Input:** subtitle text file for one movie or episode (plain `.txt`, dialogue only, cleaned and preprocessed in previous steps).
- **Output:** a fixed-length extractive summary made using the highest-scoring subtitle segments, kept in their original order.
- **Goal:** test how much of a plot a model can recover just by using the subtitle text file.


### spaCy usage

spaCy (`en_core_web_sm`) is used for [tokenization / sentence splitting]. 

## Baselines and comparisons

All systems produce summaries of the same fixed length.

| System | Description |
|---|---|
| Random | Random units until the length limit (fixed seed) |
| Lead | First units of the transcript until the length limit |
| TF-IDF unigram | `ngram_range=(1, 1)` |
| TF-IDF unigram + bigram | `ngram_range=(1, 2)` |

## Repository structure

```
.
├── README.md
├── requirements.txt
├── [summarizer.py]        # unit splitting, TF-IDF scoring, summary selection
├── [baselines.py]         # random and lead baselines
├── [evaluate.py]          # ROUGE evaluation and results table
├── [top_bigrams.py]       # high-weight bigram output
├── outputs/               # sample generated summaries
└── data/                  # NOT committed (see below)
    ├── transcripts/       # subtitle .txt files
    └── references.csv     # title, filename, reference_summary
```

## Software

Python, pandas, scikit-learn, spaCy, rouge-score.
