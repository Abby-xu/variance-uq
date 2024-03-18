import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import uncertainty
from jaxtyping import Float, Int, Bool, Str
from torch import Tensor
from utils import data_utils, logging_utils
from utils import *

def calculate_uncertainty_from_results(results: Str[Tensor, 'n_samples, n_perturb, n_responses, embedding_dim'], uncertainty_type: Str) -> Float[Tensor, 'n_samples']:
    '''
    # TODO: add semantic entropy
    '''
    embeddings = data_utils.get_all_embeddings(results)
    uncertainty_results = {}
    model_uncertainty = uncertainty.embedding_variance.calculate_model_uncertainty(embeddings)
    data_uncertainty = uncertainty.embedding_variance.calculate_data_uncertainty(embeddings)
    total_uncertainty = uncertainty.embedding_variance.calculate_total_uncertainty(embeddings)
    for idx in range(results.size(0)):
        sample_results = results[idx].flatten()
        exact_match_entropy = uncertainty.calculate_exact_match_entropy(sample_results)
        rouge_l_uncertainty = uncertainty.calculate_rouge_l_uncertainty(sample_results)
        uncertainty_results[idx] = {'model_uncertainty': model_uncertainty[idx], 'data_uncertainty': data_uncertainty[idx], 'total_uncertainty': total_uncertainty[idx], 'exact_match_entropy': exact_match_entropy, 'rouge_l_uncertainty': rouge_l_uncertainty}
    return uncertainty_results