import torch
from utils import data_utils
from typing import List



def calculate_embedding_variance_from_embeddings(sample: torch.Tensor) -> torch.Tensor:
    '''
    sample: torch.Tensor of shape (n_sample, embedding_dim)
    '''
    return torch.var(sample, dim=1, unbiased=False)

def calculate_embedding_variance_from_text(responses: List[str]):
    embeddings = data_utils.get_cls_embedding(responses)
    uncertainty = calculate_embedding_variance_from_embeddings(embeddings)
    return uncertainty

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
