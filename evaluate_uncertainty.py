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

def project_embeddings(original_embeddings: Float[Tensor, 'n_sample embedding_dim'], response_embeddings: Float[Tensor, 'n_sample n_perturb n_responses embedding_dim']) -> Float[Tensor, 'n_sample n_perturb n_responses embedding_dim']:
    n_samples, n_perturb, n_responses, embedding_dim = response_embeddings.shape
    
    results = torch.zeros(n_samples, n_perturb, n_responses, embedding_dim)
    for i in range(n_samples):
        for j in range(n_perturb):
            for k in range(n_responses):
                results[i, j, k] = original_embeddings[i]

def evaluate_uncertainty(responses, tokenizer: AutoTokenizer, model: AutoModel, args, original_questions, concat_responses=False) -> Float[Tensor, 'n_samples']:
    '''
    results: Array[str, 'n_samples n_perturb n_responses'])
    # TODO: add semantic entropy
    '''

    classify_wrapper = uncertainty.ClassifyWrapper()

    if concat_responses:
        new_responses = np.vectorize(concatenate_strings)(original_questions, responses)
        embeddings = data_utils.get_all_embeddings(new_responses, model, tokenizer, args)
        # original_questions_temp = original_questions[:, :, 0, 0]
        # original_question_embeddings = data_utils.get_all_embeddings(original_questions_temp, model, tokenizer, args)
        # response_embeddings = data_utils.get_all_embeddings(responses, model, tokenizer, args)
        # # project response embeddings onto the same space as original question embeddings
        # embeddings = project_embeddings(original_question_embeddings, response_embeddings)

    else:
        embeddings = data_utils.get_all_embeddings(responses, model, tokenizer, args)

    
    uncertainty_results = {}
    model_uncertainty = uncertainty.embedding_variance.calculate_model_uncertainty_from_results(embeddings)
    data_uncertainty = uncertainty.embedding_variance.calculate_data_uncertainty_from_results(embeddings)
    total_uncertainty = uncertainty.embedding_variance.calculate_total_uncertainty_from_results(embeddings)
    # pca_uncertainty = uncertainty.embedding_variance.calculate_pca_variance_from_embeddings(embeddings)
    # if original_questions is not None:
    #     aligned_uncertainty = uncertainty.embedding_variance.calculate_question_aligned_variance_from_embeddings(question_embeddings, embeddings)
    #     uncertainty_results['aligned_uncertainty'] = aligned_uncertainty
    
    uncertainty_results['model_uncertainty'] = model_uncertainty
    uncertainty_results['data_uncertainty'] = data_uncertainty
    uncertainty_results['total_uncertainty'] = total_uncertainty
    # uncertainty_results['pca_uncertainty'] = pca_uncertainty
    

    exact_match_uncertainty = torch.zeros(responses.shape[0])
    rouge_l_uncertainty = torch.zeros(responses.shape[0])
    # num_sets_uncertainty = torch.zeros(responses.shape[0])
    semantic_entropy_uncertainty = torch.zeros(responses.shape[0])
    eigv_uncertainty = torch.zeros(responses.shape[0])
    ecc_uncertainty = torch.zeros(responses.shape[0])

    for idx in range(responses.shape[0]):
        sample_results = responses[idx].flatten()
        original_question = original_questions[idx, 0, 0]
        exact_match_uncertainty[idx] = uncertainty.calculate_exact_match_entropy(sample_results)
        rouge_l_uncertainty[idx] = uncertainty.calculate_rouge_l_uncertainty(sample_results)
        # num_sets_uncertainty[idx] = uncertainty.calculate_num_semantic_sets(original_question, sample_results)
        semantic_entropy_uncertainty[idx] = uncertainty.calculate_semantic_entropy(original_question, sample_results, classify_wrapper)
        eigv_uncertainty[idx] = uncertainty.calculate_eigv(original_question, sample_results, classify_wrapper)
        ecc_uncertainty[idx] = uncertainty.calculate_ecc(original_question, sample_results, classify_wrapper)
        print('Semantic Entropy:', semantic_entropy_uncertainty[idx])
        print('Eigv:', eigv_uncertainty[idx])
        print('Ecc:', ecc_uncertainty[idx])

    uncertainty_results['exact_match_uncertainty'] = exact_match_uncertainty
    uncertainty_results['rouge_l_uncertainty'] = rouge_l_uncertainty
    # uncertainty_results['num_sets_uncertainty'] = num_sets_uncertainty
    uncertainty_results['semantic_entropy_uncertainty'] = semantic_entropy_uncertainty
    uncertainty_results['eigv_uncertainty'] = eigv_uncertainty
    uncertainty_results['ecc_uncertainty'] = ecc_uncertainty

    return uncertainty_results