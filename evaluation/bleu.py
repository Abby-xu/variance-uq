from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, hypothesis):
    """
    Calculate the BLEU score between a reference sentence and a hypothesis sentence.

    Args:
        reference (list): Reference sentence as a list of tokens.
        hypothesis (list): Hypothesis sentence as a list of tokens.

    Returns:
        float: BLEU score between the reference and hypothesis sentences.
    """
    return sentence_bleu([reference], hypothesis)


if __name__ == '__main__':
    # Example usage
    reference = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    hypothesis = ["The", "fast", "brown", "fox", "jumps", "over", "the", "sleepy", "dog"]
    bleu_score = calculate_bleu(reference, hypothesis)
    print("BLEU score:", bleu_score)