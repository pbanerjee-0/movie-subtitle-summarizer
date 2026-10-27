from rouge_score import rouge_scorer

def evaluate_summarizer(generated_text, reference_text):
    # Initialize the ROUGE scorer for unigrams (rouge1), bigrams (rouge2), and Longest Common Subsequence (rougeL)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(reference_text, generated_text)
    
    return scores

if __name__ == "__main__":
    # Example strings (replace  with actual model output and reference text)
    generated_summary = "Agatha Harkness. You have betrayed your coven. What if we summon a backup Green Witch?"
    reference_summary = "Agatha is accused of betraying her coven by the Salem Seven, prompting the group to consider summoning a backup Green Witch."
    
    results = evaluate_summarizer(generated_summary, reference_summary)
    
    print("--- Baseline ROUGE Evaluation Results ---")
    print(f"ROUGE-1 F1 Score: {results['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2 F1 Score: {results['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L F1 Score: {results['rougeL'].fmeasure:.4f}")