import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import uncertainty
import evaluation
from typing import Dict
from jaxtyping import Float, Int, Bool
from torch import Tensor
from utils import *

def evaluate_accuracy(
        questions,
        results,
        answers,
        args,
) -> Dict[str, Bool[Tensor, 'n_samples n_perturb n_responses']]:
    '''
    questions: Array[str, 'n_samples n_perturb n_responses embedding_dim']
    results: Array[str, 'n_samples n_perturb n_responses embedding_dim']
    answers: Array[str, 'n_samples n_perturb n_responses embedding_dim']
    '''
    accuracy_results = {
        'exact_match': Bool[Tensor, 'n_samples n_perturb n_responses'],
        'ask_for_accuracy': Bool[Tensor, 'n_samples n_perturb n_responses'],
    }

    exact_match_accuracy_results = torch.zeros(*results.shape)
    ask_for_accuracy_results = torch.zeros(*results.shape)

    for sample_idx in range(results.shape[0]):
        for perturb_idx in range(results.shape[1]):
            for response_idx in range(results.shape[2]):
                # for perturbations where the answer does not change, answer array will be the same for each sample_idx, so really this array is 1D, but we repeat it to match the shape of results
                # may be useful to have different answers for different perturbations in the future
                question = questions[sample_idx, perturb_idx, response_idx]
                answer = answers[sample_idx, perturb_idx, response_idx]
                response = results[sample_idx, perturb_idx, response_idx]
                exact_match = evaluation.calculate_exact_match(response, answer)
                ask_for_accuracy = evaluation.calculate_ask_for_accuracy(question, response, answer, args)
                exact_match_accuracy_results[sample_idx, perturb_idx, response_idx] = exact_match
                ask_for_accuracy_results[sample_idx, perturb_idx, response_idx] = ask_for_accuracy
    
    accuracy_results['exact_match'] = exact_match_accuracy_results
    accuracy_results['ask_for_accuracy'] = ask_for_accuracy_results
    return accuracy_results
                