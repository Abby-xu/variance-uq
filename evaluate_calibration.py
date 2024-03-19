from sklearn.metrics import roc_auc_score
from typing import List, Dict
from torch import Tensor
from jaxtyping import Float, Bool
import numpy as np
import einops
from tabulate import tabulate
from matplotlib import pyplot as plt

def evaluate_calibration(
    uncertainty_results: Dict[str, Float[Tensor, 'n_samples n_perturb n_responses']],
    accuracy_results: Dict[str, Bool[Tensor, 'n_samples n_perturb n_responses']],
) -> Dict[str, Float]:
    confidence_metrics = 'rouge_l_uncertainty'
    
    # Create a dictionary to store the table data
    table_data = {}
    
    # go through each combination of uncertainty and accuracy
    roc_auc_results = {}
    for uncertainty_type, uncertainty_tens in uncertainty_results.items():
        if uncertainty_type not in confidence_metrics:
            uncertainty_tens = 1 - uncertainty_tens
        
        if uncertainty_type not in table_data:
            table_data[uncertainty_type] = {}
        
        for accuracy_type, accuracy_tens in accuracy_results.items():
            roc_auc = calculate_roc_auc(uncertainty_tens, accuracy_tens)
            roc_auc_results[f'{uncertainty_type}_{accuracy_type}'] = roc_auc
            
            # Store the result in the table_data dictionary
            table_data[uncertainty_type][accuracy_type] = f"{roc_auc:.3}"
    
    # Convert the table_data dictionary to a list of lists for tabulate
    table_headers = ["Uncertainty Type"] + list(accuracy_results.keys())
    table_rows = []
    for uncertainty_type, accuracy_results in table_data.items():
        row = [uncertainty_type] + [accuracy_results.get(acc_type, "-") for acc_type in table_headers[1:]]
        table_rows.append(row)
    
    # Print the table
    print("Calibration Evaluation Results:")
    print(tabulate(table_rows, headers=table_headers, tablefmt="grid"))
    print()

    # TODO: create ROC plots

    # # Create bar plots
    # for accuracy_type in accuracy_results.keys():
    #     plt.figure(figsize=(8, 6))
    #     uncertainty_types = list(table_data.keys())
    #     accuracy_values = [float(table_data[unc_type][accuracy_type]) for unc_type in uncertainty_types]
        
    #     plt.bar(uncertainty_types, accuracy_values)
    #     plt.xlabel("Uncertainty Type")
    #     plt.ylabel("Accuracy")
    #     plt.title(f"Accuracy vs. Uncertainty ({accuracy_type})")
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()
    
    return roc_auc_results

def calculate_roc_auc(
        uncertainty_tens: Float[Tensor, 'n_samples n_perturb n_responses'],
        accuracy_tens: Float[Tensor, 'n_samples n_perturb n_responses'],
) -> Dict[str, Float]:
    '''
    Use uncertainty values to predict accuracy
    '''
    # We only collect uncertainty values over each sample, so we need to repeat the uncertainty values for each perturbation and response
    # TODO: in the future, we might actually want to look at the unique uncertainty values under each perturbation.
    uncertainty_tens = einops.repeat(uncertainty_tens, 'n_samples -> n_samples n_perturb n_responses', n_perturb=accuracy_tens.shape[1], n_responses=accuracy_tens.shape[2])
    uncertainty_vals = uncertainty_tens.flatten().detach().cpu().numpy()
    accuracy_vals = accuracy_tens.flatten().detach().cpu().numpy()
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