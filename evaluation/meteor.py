from nltk.translate.meteor_score import single_meteor_score

def calculate_meteor(reference, hypothesis):
    """
    Calculate the METEOR score between a reference sentence and a hypothesis sentence.

    Args:
        reference (str): Reference sentence as a string.
        hypothesis (str): Hypothesis sentence as a string.

    Returns:
        float: METEOR score between the reference and hypothesis sentences.
    """
    return single_meteor_score(reference, hypothesis)

if __name__ == '__main__':
    # Example usage
    reference = "The quick brown fox jumps over the lazy dog"
    hypothesis = "The fast brown fox jumps over the sleepy dog"
    meteor_score = calculate_meteor(reference, hypothesis)
    print("METEOR score:", meteor_score)