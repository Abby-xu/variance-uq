import argparse
import os
import openai
from models import get_model, get_tokenizer
from uncertainty import embedding_variance, rouge_l, semantic_entropy
from evaluation import bleu, meteor, bertscore
import config
import numpy as np
import pandas as pd
from utils import data_utils, logging_utils
import pickle
from typing import List
from generate_responses import generate_responses
from evaluate_uncertainty import evaluate_uncertainty
from evaluate_accuracy import evaluate_accuracy
from evaluate_calibration import evaluate_calibration
import einops

openai.api_key = os.environ['OPENAI_API_KEY']

def experiment(args, preload_results=False, save_results=True):
    if preload_results:
        pass
    else:
        results = generate_responses(args)

    uncertainty_results = evaluate_uncertainty(results, args) # dict of tensors of shape (n_samples)
    accuracy_results = evaluate_accuracy(results, args) # dict of tensors of shape (n_samples, n_perturb, n_responses)
    calibration_results = evaluate_calibration(uncertainty_results, accuracy_results) # dict of roc_auc values, each key refers to the pair of uncertainty and accuracy metrics

    if save_results:
        timestamp = logging_utils.get_timestamp()
        results_save_filename = f'results_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        results_save_filepath = os.path.join('results', results_save_filename)
        os.makedirs(os.path.dirname(results_save_filepath), exist_ok=True)
        with open(results_save_filepath, 'wb') as f:
            pickle.dump(results, f)

        uncertainty_save_filename = f'uncertainty_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        uncertainty_save_filepath = os.path.join('results', uncertainty_save_filename)
        os.makedirs(os.path.dirname(uncertainty_save_filepath), exist_ok=True)
        with open(uncertainty_save_filepath, 'wb') as f:
            pickle.dump(uncertainty_results, f)

        accuracy_save_filename = f'accuracy_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        accuracy_save_filepath = os.path.join('results', accuracy_save_filename)
        os.makedirs(os.path.dirname(accuracy_save_filepath), exist_ok=True)
        with open(accuracy_save_filepath, 'wb') as f:
            pickle.dump(accuracy_results, f)

        calibration_save_filename = f'calibration_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        calibration_save_filepath = os.path.join('results', calibration_save_filename)
        os.makedirs(os.path.dirname(calibration_save_filepath), exist_ok=True)
        with open(calibration_save_filepath, 'wb') as f:
            pickle.dump(calibration_results, f)

    return results, uncertainty_results, accuracy_results, calibration_results

# def experiment(args, preload=True, save_results=True):
#     if not preload:
#         tokenizer, model = config.initialize_components(args)
        
#         results = []
#         dataset = data_utils.load_dataset(args)
#         for idx_test, data in enumerate(dataset):
#             if args.dataset == 'trivia_qa':
#                 input = data['input']
#                 answer = data['answer']['aliases'][0]
#             elif args.dataset == 'geneset':
#                 input = data['input'].split(' ')
#                 answer = data['answer']
#             else:
#                 raise NotImplementedError
#             example_results = {'idx': idx_test, 'results': [], 'answer': answer}
#             for idx_perturb in range(args.n_perturb):
#                 perturb_results = {'idx': idx_perturb, 'results': []}
#                 perturbed_sample = data_utils.perturb_sample(input, args)
#                 print()
#                 print('Input:', input)
#                 print('Prompt:', perturbed_sample)
#                 if args.dataset == 'geneset':
#                     prompt = data_utils.geneset_sample_to_prompt(perturbed_sample)
#                 else:
#                     prompt = data_utils.sample_to_prompt(perturbed_sample)
#                 responses = data_utils.generate_response(prompt, args)
#                 for idx_sample, response in enumerate(responses):
#                     if response == '':
#                         rouge_l_score = 0
#                         exact_match = 0
#                         correct_per_gpt = 0
#                     else:
#                         print('Response:', response)
#                         rouge_l_score = rouge_l.calculate_rouge_l(response, answer)
#                         exact_match = data_utils.exact_match(response, answer)
#                         correct_per_gpt = int(data_utils.prompt_correctness(input, response, answer, args))
#                     perturb_results['results'].append({'idx': idx_sample, 'response': response, 'rouge_l_score': rouge_l_score, 'exact_match': exact_match, 'correct_per_gpt': correct_per_gpt})
#                 example_results['results'].append(perturb_results)
#             results.append(example_results)

#         embeddings = data_utils.get_all_embeddings(results, model, tokenizer, args)
        
#         model_uncertainty = embedding_variance.calculate_model_uncertainty(embeddings)
#         data_uncertainty = embedding_variance.calculate_data_uncertainty(embeddings)
#         total_uncertainty = embedding_variance.calculate_total_uncertainty(embeddings)
        
#         uncertainty_decomposition_results = {'model_uncertainty': model_uncertainty, 'data_uncertainty': data_uncertainty, 'total_uncertainty': total_uncertainty}
#         if save_results:
#             save_filename = 'results_{}_{}_{}_{}.pkl'.format(args.dataset, args.split, args.model, args.uq_metric)
#             save_filepath = os.path.join('results', save_filename)
#             os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
#             with open(save_filepath, 'wb') as f:
#                 pickle.dump(results, f)
#             with open(save_filepath.replace('.pkl', '_uncertainty_decomposition.pkl'), 'wb') as f:
#                 pickle.dump(uncertainty_decomposition_results, f)
#     else:
#         filename = '/Users/kylecox/Documents/ws/variance-uq/results/results_trivia_qa_validation_gpt-3.5-turbo_all.pkl'
#         with open(filename, 'rb') as f:
#             results = pickle.load(f)

#         filename = '/Users/kylecox/Documents/ws/variance-uq/results/results_trivia_qa_validation_gpt-3.5-turbo_all_uncertainty_decomposition.pkl'
#         with open(filename, 'rb') as f:
#             uncertainty_decomposition_results = pickle.load(f)

#     roc_auc_results = data_utils.calculate_roc_auc(results, uncertainty_decomposition_results)
#     print(roc_auc_results)
#     return roc_auc_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation of the model")
    parser.add_argument("--dataset", type=str, default="trivia_qa", choices=['trivia_qa', 'geneset'], help="The dataset to use")
    parser.add_argument("--split", type=str, default="validation", help="The split to evaluate on")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model to use")
    parser.add_argument("--n_test", type=int, default=200, help="The number of validation samples to test")
    parser.add_argument("--n_perturb", type=int, default=20, help="The number of perturbed samples to test for each validation example")
    parser.add_argument("--n_sample", type=int, default=5, help="The number of samples to generate for each prompt")
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2", help="The embedding model to use")
    parser.add_argument("--temperature", type=float, default=1, help="The temperature to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="The maximum number of tokens to generate")
    parser.add_argument("--stop", type=str, default=None, help="The stop sequence")
    parser.add_argument("--uq_metric", type=str, default="all", choices=['variance', 'entropy_exact', 'entropy_cluster', 'all'], help="The uncertainty metric to use")
    parser.add_argument("--perturb_type", type=str, default="rephrase", choices=["bootstrap", "permute"], help="The type of perturbation to use")

    args = parser.parse_args()
    
    experiment(args)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run the evaluation of the model")
#     parser.add_argument("--dataset", type=str, default="trivia_qa", choices=['trivia_qa', 'geneset'], help="The dataset to use")
#     parser.add_argument("--split", type=str, default="validation", help="The split to evaluate on")
#     parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model to use")
#     parser.add_argument("--n_test", type=int, default=200, help="The number of validation samples to test")
#     parser.add_argument("--n_perturb", type=int, default=20, help="The number of perturbed samples to test for each validation example")
#     parser.add_argument("--n_sample", type=int, default=5, help="The number of sampels to generate for each prompt")
#     parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2", help="The embedding model to use")
#     parser.add_argument("--temperature", type=float, default=1, help="The temperature to use")
#     parser.add_argument("--max-tokens", type=int, default=100, help="The maximum number of tokens to generate")
#     parser.add_argument("--stop", type=str, default=None, help="The stop sequence")
#     parser.add_argument("--uq_metric", type=str, default="all", choices=['variance', 'entropy_exact', 'entropy_cluster', 'all'], help="The uncertainty metric to use")
#     parser.add_argument("--perturb_type", type=str, default="rephrase", choices=["bootstrap", "permute"], help="The type of perturbation to use")

#     args = parser.parse_args()
    
#     experiment(args)
    