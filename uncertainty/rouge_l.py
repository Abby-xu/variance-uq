from rouge import Rouge
from typing import List
from torch import Tensor
from jaxtyping import Float
import numpy as np

def calculate_rouge_l(outputs: Float[Tensor, 'n_perturb * n_sample'], answer: str) -> float:
    '''
    Calculates the rouge-l score between the outputs and the answer
    '''
    
    outputs = outputs.detach().cpu().tolist()
    rouge = Rouge()
    scores = []
    for output in outputs:
        score = rouge.get_scores(output, answer, avg=True)
        scores.append(score['rouge-l']['f'])
    return sum(scores) / len(scores)

def calculate_rouge_l_uncertainty(outputs: np.array) -> float:
    '''
    outputs: Float[Tensor, 'n_perturb * n_sample'])
    Calculates the average rouge-l score between responses
    '''

    rouge = Rouge()
    scores = []

    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            score = rouge.get_scores(outputs[i], outputs[j], avg=True)
            scores.append(score['rouge-l']['f'])

    if len(scores) > 0:
        average_score = sum(scores) / len(scores)
    else:
        average_score = 0.0

    return average_score