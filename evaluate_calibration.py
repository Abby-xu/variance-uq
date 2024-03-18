from sklearn.metrics import roc_auc_score
from typing import List, Str
from torch import Tensor
from jaxtyping import Str, Dict, Float
import numpy as np

def calculate_roc_auc_all_results(
        results: Str[Tensor, 'n_samples, n_perturb, n_responses'],
        uncertainty_results: Dict[Str, Str[Tensor, 'n_samples, n_perturb, n_responses']],
        accuracy_results: Dict[Str, Str[Tensor, 'n_samples, n_perturb, n_responses']],
) -> Dict[Str, Str, Float]:
    
    # go through each combination of uncertainty and accuracy
    roc_auc_results = {}
    for uncertainty_type, uncertainty_tens in uncertainty_results.items():
        for accuracy_type, accuracy_tens in accuracy_results.items():
            roc_auc = calculate_roc_auc(uncertainty_tens, accuracy_tens)
            roc_auc_results[f'{uncertainty_type}_{accuracy_type}'] = roc_auc
    return roc_auc_results

def calculate_roc_auc(
        uncertainty_results: Str[Tensor, 'n_samples, n_perturb, n_responses'],
        accuracy_results: Str[Tensor, 'n_samples, n_perturb, n_responses'],
) -> Dict[Str, Str, Float]:
    '''
    Use uncertainty values to predict accuracy
    '''

    uncertainty_vals = uncertainty_results.flatten().detach().cpu().numpy()
    accuracy_vals = accuracy_results.flatten().detach().cpu().numpy()
    roc_auc = roc_auc_score(accuracy_vals, uncertainty_vals)
    return roc_auc

# def calculate_roc_auc(results, uncertainty_decomposition_results):
#     # use the total uncertainty to calculate the roc auc
#     total_var_uq_vals = []
#     model_var_uq_vals = []
#     data_var_uq_vals = []
#     entropy_uq_vals = []
#     lexi_sim_uq_vals = []
#     num_distinct_answers_uq_vals = []
#     exact_match_vals = []
#     rouge_l_vals = []
#     correct_per_gpt_vals = []

#     for idx, example_results in enumerate(results):
#         total_uncertainty = uncertainty_decomposition_results['total_uncertainty'][idx].item()
#         model_uncertainty = uncertainty_decomposition_results['model_uncertainty'][idx].item()
#         data_uncertainty = uncertainty_decomposition_results['data_uncertainty'][idx].item()
#         outputs = []
#         answer = None
#         answer = example_results['answer']
#         for idx_perturb, perturb_results in enumerate(example_results['results']):
#             for idx_sample, sample_result in enumerate(perturb_results['results']):
#                 if sample_result['response'] == '':
#                     output = ' '
#                 else:
#                     output = sample_result['response']
#                 outputs.append(output)
#                 exact_match = int(sample_result['exact_match'])
#                 rouge_l = sample_result['rouge_l_score']
#                 rouge_l = int(rouge_l > 0.3)
#                 correct_per_gpt = sample_result['correct_per_gpt']

#                 # add accuracy metrics
#                 exact_match_vals.append(exact_match)
#                 rouge_l_vals.append(rouge_l)
#                 correct_per_gpt_vals.append(correct_per_gpt)
                
#                 # add var uq metrics
#                 total_var_uq_vals.append(total_uncertainty)
#                 model_var_uq_vals.append(model_uncertainty)
#                 data_var_uq_vals.append(data_uncertainty)

#         # calculate exact match entropy
#         exact_match_entropy = calculate_entropy_exact(outputs)
#         entropy_uq_vals.extend([exact_match_entropy] * len(perturb_results['results']) * len(example_results['results']))

#         # calculate number of distinct answers
#         num_distinct_answers_uq_vals.extend([len(set(outputs))] * len(perturb_results['results']) * len(example_results['results']))

#         # calculate lexi sim entropy
#         lexi_sim = calculate_lexical_similarity(outputs)
#         lexi_sim_uq_vals.extend([lexi_sim] * len(perturb_results['results']) * len(example_results['results']))

#     total_var_uq_vals = torch.tensor(total_var_uq_vals)
#     entropy_uq_vals = torch.tensor(entropy_uq_vals)
#     exact_match_vals = torch.tensor(exact_match_vals)

#     uq_metrics = {'total_variance': total_var_uq_vals, 'model_variance': model_var_uq_vals, 'total_variance': total_var_uq_vals, 'entropy_exact': entropy_uq_vals, 'lexi_sim': lexi_sim_uq_vals, 'num_distinct_answers': num_distinct_answers_uq_vals}
#     accuracy_metrics = {'correct_per_gpt': correct_per_gpt_vals}

#     roc_auc_results = []
#     for uq_metric, uq_vals in uq_metrics.items():
#         # if uq_vals is tensor, convert to numpy array
#         if isinstance(uq_vals, torch.Tensor):
#             uq_vals = uq_vals.detach().numpy()
#         for accuracy_metric, accuracy_vals in accuracy_metrics.items():
#             if isinstance(accuracy_vals, torch.Tensor):
#                 accuracy_vals = accuracy_vals.detach().numpy()
#             print(f'{uq_metric}: {np.mean(uq_vals):.3}')
#             print(f'{accuracy_metric}: {np.mean(accuracy_vals):.3}')
#             if uq_metric == 'lexi_sim':
#                 roc_auc = roc_auc_score(accuracy_vals, uq_vals)
#             else:
#                 roc_auc = 1 - roc_auc_score(accuracy_vals, uq_vals)
#             roc_auc_results.append({'uq_metric': uq_metric, 'accuracy_metric': accuracy_metric, 'roc_auc': roc_auc})
#             print(f'AUROC: {roc_auc:.3}')
#             print()

#     return roc_auc_results