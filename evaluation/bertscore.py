from bert_score import score

def calculate_bertscore(references, hypotheses):
    """
    Calculate the BERTScore between a list of reference sentences and a list of hypothesis sentences.

    Args:
        references (list): List of reference sentences as strings.
        hypotheses (list): List of hypothesis sentences as strings.

    Returns:
        tuple: Tuple containing the precision, recall, and F1 scores.
    """
    precision, recall, f1 = score(hypotheses, references, lang='en', verbose=False)
    return float(precision.mean()), float(recall.mean()), float(f1.mean())

if __name__ == '__main__':
    # Example usage
    references = [
        "The quick brown fox jumps over the lazy dog",
        "The cat sits on the mat"
    ]
    hypotheses = [
        "The fast brown fox jumps over the sleepy dog",
        "The cat is sitting on the mat"
    ]
    precision, recall, f1 = calculate_bertscore(references, hypotheses)
    print("BERTScore - Precision:", precision)
    print("BERTScore - Recall:", recall)
    print("BERTScore - F1:", f1)