from rouge_score import rouge_scorer

def evaluate_summarizer(generated_text, reference_text):
    # Initialize the ROUGE scorer for unigrams (rouge1), bigrams (rouge2), and Longest Common Subsequence (rougeL)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(reference_text, generated_text)
    
    return scores

if __name__ == "__main__":
    # Read TF-IDF generated summary file
    with open('data/output_summaries/black_widow_2021_summary.txt', 'r', encoding='utf-8') as f:
        generated_summary = f.read().strip()
        
    # Read reference summary file
    with open('data/ref_summaries/black_widow_2021_original.txt', 'r', encoding='utf-8') as f:
        reference_summary = f.read().strip()
    
    # Score and evaluate with ROUGE
    results = evaluate_summarizer(generated_summary, reference_summary)
    
    print("--- Baseline ROUGE Evaluation Results ---")
    print(f"ROUGE-1 F1 Score: {results['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2 F1 Score: {results['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L F1 Score: {results['rougeL'].fmeasure:.4f}")