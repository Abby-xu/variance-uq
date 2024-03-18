import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import uncertainty
import evaluation
from jaxtyping import Float, Int, Bool, Str, Dict
from torch import Tensor
from utils import data_utils, logging_utils
from utils import *

def calculate_correctness_from_results(
        results: Str[Tensor, 'n_samples, n_perturb, n_responses, embedding_dim'], 
        answers: Str[str, 'n_samples'],
) -> Bool[Str, Float[Tensor, 'n_samples, n_perturb, n_responses']]:
    '''

    '''
    accuracy_results = {
        'exact_match': Bool[Tensor, 'n_samples, n_perturb, n_responses'],
        'ask_for_accuracy': Bool[Tensor, 'n_samples, n_perturb, n_responses'],
    }

    exact_match_accuracy_results = torch.zeros(results.size(0), results.size(1), results.size(2))
    ask_for_accuracy_results = torch.zeros(results.size(0), results.size(1), results.size(2))

    for sample_idx in range(results.size(0)):
        answer = answers[sample_idx]
        for perturb_idx in range(results.size(1)):
            for response_idx in range(results.size(2)):
                exact_match = evaluation.calculate_exact_match(results[sample_idx, perturb_idx, response_idx], answer)
                ask_for_accuracy = evaluation.calculate_ask_for_accuracy(results[sample_idx, perturb_idx, response_idx], answer)
                exact_match_accuracy_results[sample_idx, perturb_idx, response_idx] = exact_match
                ask_for_accuracy_results[sample_idx, perturb_idx, response_idx] = ask_for_accuracy
    
    accuracy_results['exact_match'] = exact_match_accuracy_results
    accuracy_results['ask_for_accuracy'] = ask_for_accuracy_results
    return accuracy_results
                