import torch
import numpy as np
from sklearn.metrics import roc_auc_score

import uncertainty
from jaxtyping import Float, Int, Bool
from torch import Tensor
from utils import data_utils, logging_utils
from utils import *
from transformers import AutoTokenizer, AutoModel

def concatenate_strings(str1, str2):
    return str1 + ' ' + str2 + '.'

def evaluate_uncertainty(responses, tokenizer: AutoTokenizer, model: AutoModel, args, original_questions = None) -> Float[Tensor, 'n_samples']:
    '''
    results: Array[str, 'n_samples n_perturb n_responses'])
    # TODO: add semantic entropy
    '''
    if original_questions is not None:
        # original questions and results are arrays of the same shape, each containing strings. We want to concatenate them together, with a space in between.
        responses = np.vectorize(concatenate_strings)(original_questions, responses)
    embeddings = data_utils.get_all_embeddings(responses, model, tokenizer, args)
    print(responses)
    # if original_questions is not None:
    #     original_question_embeddings = data_utils.get_all_embeddings(original_questions, model, tokenizer, args)
    #     # project the embeddings to the same space
        

    uncertainty_results = {}
    model_uncertainty = uncertainty.embedding_variance.calculate_model_uncertainty_from_results(embeddings)
    data_uncertainty = uncertainty.embedding_variance.calculate_data_uncertainty_from_results(embeddings)
    total_uncertainty = uncertainty.embedding_variance.calculate_total_uncertainty_from_results(embeddings)
    
    uncertainty_results['model_uncertainty'] = model_uncertainty
    uncertainty_results['data_uncertainty'] = data_uncertainty
    uncertainty_results['total_uncertainty'] = total_uncertainty

    exact_match_uncertainty = torch.zeros(responses.shape[0])
    rouge_l_uncertainty = torch.zeros(responses.shape[0])

    for idx in range(responses.shape[0]):
        sample_results = responses[idx].flatten()
        exact_match_uncertainty[idx] = uncertainty.calculate_exact_match_entropy(sample_results)
        rouge_l_uncertainty[idx] = uncertainty.calculate_rouge_l_uncertainty(sample_results)

    uncertainty_results['exact_match_uncertainty'] = exact_match_uncertainty
    uncertainty_results['rouge_l_uncertainty'] = rouge_l_uncertainty

    return uncertainty_results