import torch
from torch import Tensor
from utils import data_utils
from typing import List
from jaxtyping import Float
import numpy as np

# TODO: fix type hinting in this file, we are taking arrays instead of tensors for results

def calculate_question_aligned_variance_from_embeddings(question_embeddings: Float[Tensor, 'n_sample n_perturb n_responses embedding_dim'], embeddings: Float[Tensor, 'n_sample n_perturb n_responses embedding_dim']) -> Float[Tensor, 'n_sample']:
    '''
    For each answer embedding, calculates its cosine similarity with the question embedding, and then weights the variance of the answer embeddings by the cosine similarity
    '''
    weighted_variances = []
    # print(question_embeddings.shape)
    question_embeddings = question_embeddings[:, 0, 0, :]
    flattened_embeddings = embeddings.flatten(-3, -2)
    for i, sample_embeddings in enumerate(flattened_embeddings):
        question_embedding = question_embeddings[i].unsqueeze(0)
        print(question_embedding.shape)
        print(sample_embeddings.shape)
        similarities = torch.nn.functional.cosine_similarity(question_embedding, sample_embeddings, dim=1)
        weighted_variance = (sample_embeddings.var(dim=0) * similarities).sum()
        weighted_variances.append(weighted_variance)
    weighted_variances = torch.stack(weighted_variances)
    return weighted_variances



def calculate_pca_variance_from_embeddings(embeddings: Float[Tensor, 'n_sample n_perturb n_responses embedding_dim'], n_components=100) -> Float[Tensor, 'n_sample']:
    '''
    Takes the embeddings for each test example (i.e. flattening along the n_perturb and n_responses axis) and calculates the principal components of the embeddings
    Then projects the embeddings onto the principal components and calculates the variance of the projected embeddings
    '''


    # if embeddings have 4 dimensions, flatten the first two dimensions
    if len(embeddings.shape) == 4:
        flattened_embeddings = embeddings.flatten(0, 1)
    else:
        flattened_embeddings = embeddings
    # calculate pca for each sample
    pca_variances = []
    if len(flattened_embeddings.shape) == 2:
        flattened_embeddings = flattened_embeddings.unsqueeze(0)
    for sample_embeddings in flattened_embeddings:
        # center the embeddings
        centered_embeddings = sample_embeddings - sample_embeddings.mean(dim=0)
        centered_embeddings = centered_embeddings.detach().cpu().numpy()
        cov_matrix = np.cov(centered_embeddings.T)
        # compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # take only the top n components
        eigenvectors = eigenvectors[:, :n_components]

        # project embeddings onto the principal components
        projected_embeddings = np.matmul(centered_embeddings, eigenvectors)

        # TODO: are we sure we should be varying over dim 0?
        # calculate the variance of the projected embeddings
        pca_variance = np.sum(np.var(projected_embeddings, axis=0))
        pca_variances.append(pca_variance)
    
    
    return pca_variances

def calculate_embedding_variance_from_embeddings(sample: torch.Tensor) -> torch.Tensor:
    '''
    sample: torch.Tensor of shape (n_sample, embedding_dim)
    '''
    return torch.var(sample, dim=1, unbiased=False)

def calculate_embedding_variance_from_text(responses: List[str], model, tokenizer):
    responses = np.array(responses)
    embeddings = data_utils.get_cls_embeddings(responses, model, tokenizer, to_numpy=False)
    uncertainty = calculate_embedding_variance_from_embeddings(embeddings)
    return uncertainty

def calculate_pca_variance_from_text(responses: List[str], model, tokenizer, n_components=100): 
    responses = np.array(responses)
    embeddings = data_utils.get_cls_embeddings(responses, model, tokenizer, to_numpy=False)
    pca_variance = calculate_pca_variance_from_embeddings(embeddings, n_components)
    return pca_variance

def calculate_uncertainty_at_pos_from_results(sample: torch.Tensor, uncertainty_type: str) -> torch.Tensor:
    '''
    sample: torch.Tensor of shape (n_sample, n_perturbed_samples, embedding_dim)
    uncertainty_type: str, one of 'model', 'data', 'total'
    model uncertainty: variance of average output from each perturbed sample
    data uncertainty: average variance of each perturbed sample
    total uncertainty: variance of all perturbed samples
    '''
    if uncertainty_type == 'model':
        # Compute the mean across n_samples and then the variance across n_perturbed_samples
        mean_across_samples = sample.mean(dim=-1)
        uncertainty = mean_across_samples.var(dim=1, unbiased=False)
    elif uncertainty_type == 'data':
        # Compute variance for each perturbed sample across n_samples and average them
        var_across_samples = torch.var(sample, dim=2, unbiased=False)
        uncertainty = var_across_samples.mean(dim=1)
    elif uncertainty_type == 'total':
        # Flatten across n_perturbed_samples and n_samples, then compute the variance
        flattened_samples = sample.view(sample.size(0), -1)
        uncertainty = torch.var(flattened_samples, dim=1, unbiased=False)
    else:
        raise ValueError("Invalid uncertainty type specified.")
    return uncertainty

def calculate_uncertainty_from_results(embeddings: torch.Tensor, uncertainty_type: str) -> torch.Tensor:
    results = []
    for i in range(embeddings.size(-1)):
        uncertainty = calculate_uncertainty_at_pos_from_results(embeddings[..., i], uncertainty_type)
        results.append(uncertainty)
    return torch.stack(results, dim=-1).sum(dim=-1)

# Now define the specific functions for model, data, and total uncertainty
def calculate_model_uncertainty_from_results(results: torch.Tensor) -> torch.Tensor:
    return calculate_uncertainty_from_results(results, 'model')

def calculate_data_uncertainty_from_results(results: torch.Tensor) -> torch.Tensor:
    return calculate_uncertainty_from_results(results, 'data')

def calculate_total_uncertainty_from_results(results: torch.Tensor) -> torch.Tensor:
    return calculate_uncertainty_from_results(results, 'total')
