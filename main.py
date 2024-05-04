import argparse
import os
import openai
from utils import data_utils, logging_utils
import pickle
from typing import List
from generate_responses import generate_responses
from evaluate_uncertainty import evaluate_uncertainty
from evaluate_accuracy import evaluate_accuracy
from evaluate_calibration import evaluate_calibration
import config
import torch

# openai.api_key = os.environ['OPENAI_API_KEY']
device = config.device
openai.api_key = config.openai_api_key

def experiment(args):
    tokenizer, model = config.initialize_components(args)
    model = model.to(device)

    preload_responses = args.preload_responses
    preload_uncertainty = args.preload_uncertainty
    preload_accuracy = args.preload_accuracy
    preload_calibration = args.preload_calibration
    save_results = args.save_results

    results_temp_dir = 'results_temp'
    results_top_dir = 'results'
    results_sub_dir = f'{args.dataset}_{args.do_perturb}_{args.n_perturb}_{args.n_sample}'
    results_dir = os.path.join(results_top_dir, results_sub_dir)

    if preload_responses:
        original_questions_save_filename = 'original_questions.pkl'
        original_questions_save_filepath = os.path.join(results_temp_dir, original_questions_save_filename)
        with open(original_questions_save_filepath, 'rb') as f:
            original_questions = pickle.load(f)
        
        perturbed_questions_save_filename = 'perturbed_questions.pkl'
        perturbed_questions_save_filepath = os.path.join(results_temp_dir, perturbed_questions_save_filename)
        with open(perturbed_questions_save_filepath, 'rb') as f:
            perturbed_questions = pickle.load(f)

        responses_save_filename = 'responses.pkl'
        responses_save_filepath = os.path.join(results_temp_dir, responses_save_filename)
        with open(responses_save_filepath, 'rb') as f:
            responses = pickle.load(f)

        answers_save_filename = 'answers.pkl'
        answers_save_filepath = os.path.join(results_temp_dir, answers_save_filename)
        with open(answers_save_filepath, 'rb') as f:
            answers = pickle.load(f)

    else:
        original_questions, perturbed_questions, responses, answers = generate_responses(args)

    if preload_uncertainty:
        uncertainty_save_filename = 'uncertainty.pkl'
        uncertainty_save_filepath = os.path.join(results_temp_dir, uncertainty_save_filename)
        with open(uncertainty_save_filepath, 'rb') as f:
            uncertainty_results = pickle.load(f)
    else:
        uncertainty_results = evaluate_uncertainty(responses, tokenizer, model, args, original_questions=original_questions) # dict of tensors of shape (n_samples)

    if preload_accuracy:
        accuracy_save_filename = 'accuracy.pkl'
        accuracy_save_filepath = os.path.join(results_temp_dir, accuracy_save_filename)
        with open(accuracy_save_filepath, 'rb') as f:
            accuracy_results = pickle.load(f)
    else:
        # When we do ask_for_accuracy, we pass in the original question, not the perturbed question. This could be changed.
        accuracy_results = evaluate_accuracy(original_questions, responses, answers, args) # dict of tensors of shape (n_samples, n_perturb, n_responses)
    
    if preload_calibration:
        calibration_save_filename = 'calibration.pkl'
        calibration_save_filepath = os.path.join(results_temp_dir, calibration_save_filename)
        with open(calibration_save_filepath, 'rb') as f:
            calibration_results = pickle.load(f)
    else:
        calibration_results = evaluate_calibration(uncertainty_results, accuracy_results) # dict of roc_auc values, each key refers to the pair of uncertainty and accuracy metrics

    if save_results:
        timestamp = logging_utils.get_timestamp()

        original_questions_save_filename = 'original_questions.pkl'
        original_questions_temp_filepath = os.path.join(results_temp_dir, original_questions_save_filename)
        original_questions_perm_filepath = os.path.join(results_dir, original_questions_save_filename)
        os.makedirs(os.path.dirname(original_questions_temp_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(original_questions_perm_filepath), exist_ok=True)
        with open(original_questions_temp_filepath, 'wb') as f:
            pickle.dump(original_questions, f)
        with open(original_questions_perm_filepath, 'wb') as f:
            pickle.dump(original_questions, f)

        perturbed_questions_save_filename = 'perturbed_questions.pkl'
        perturbed_questions_temp_filepath = os.path.join(results_temp_dir, perturbed_questions_save_filename)
        perturbed_questions_perm_filepath = os.path.join(results_dir, perturbed_questions_save_filename)
        os.makedirs(os.path.dirname(perturbed_questions_temp_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(perturbed_questions_perm_filepath), exist_ok=True)
        with open(perturbed_questions_temp_filepath, 'wb') as f:
            pickle.dump(perturbed_questions, f)
        with open(perturbed_questions_perm_filepath, 'wb') as f:
            pickle.dump(perturbed_questions, f)

        responses_save_filename = 'responses.pkl'
        responses_temp_filepath = os.path.join(results_temp_dir, responses_save_filename)
        responses_perm_filepath = os.path.join(results_dir, responses_save_filename)
        os.makedirs(os.path.dirname(responses_temp_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(responses_perm_filepath), exist_ok=True)
        with open(responses_temp_filepath, 'wb') as f:
            pickle.dump(responses, f)
        with open(responses_perm_filepath, 'wb') as f:
            pickle.dump(responses, f)

        answers_save_filename = 'answers.pkl'
        answers_temp_filepath = os.path.join(results_temp_dir, answers_save_filename)
        answers_perm_filepath = os.path.join(results_dir, answers_save_filename)
        os.makedirs(os.path.dirname(answers_temp_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(answers_perm_filepath), exist_ok=True)
        with open(answers_temp_filepath, 'wb') as f:
            pickle.dump(answers, f)
        with open(answers_perm_filepath, 'wb') as f:
            pickle.dump(answers, f)

        uncertainty_save_filename = 'uncertainty.pkl'
        uncertainty_temp_filepath = os.path.join(results_temp_dir, uncertainty_save_filename)
        uncertainty_perm_filepath = os.path.join(results_dir, uncertainty_save_filename)
        os.makedirs(os.path.dirname(uncertainty_temp_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(uncertainty_perm_filepath), exist_ok=True)
        with open(uncertainty_temp_filepath, 'wb') as f:
            pickle.dump(uncertainty_results, f)
        with open(uncertainty_perm_filepath, 'wb') as f:
            pickle.dump(uncertainty_results, f)

        accuracy_save_filename = 'accuracy.pkl'
        accuracy_temp_filepath = os.path.join(results_temp_dir, accuracy_save_filename)
        accuracy_perm_filepath = os.path.join(results_dir, accuracy_save_filename)
        os.makedirs(os.path.dirname(accuracy_temp_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(accuracy_perm_filepath), exist_ok=True)
        with open(accuracy_temp_filepath, 'wb') as f:
            pickle.dump(accuracy_results, f)
        with open(accuracy_perm_filepath, 'wb') as f:
            pickle.dump(accuracy_results, f)

        calibration_save_filename = 'calibration.pkl'
        calibration_temp_filepath = os.path.join(results_temp_dir, calibration_save_filename)
        calibration_perm_filepath = os.path.join(results_dir, calibration_save_filename)
        os.makedirs(os.path.dirname(calibration_temp_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(calibration_perm_filepath), exist_ok=True)
        with open(calibration_temp_filepath, 'wb') as f:
            pickle.dump(calibration_results, f)
        with open(calibration_perm_filepath, 'wb') as f:
            pickle.dump(calibration_results, f)

    return responses, answers, uncertainty_results, accuracy_results, calibration_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation of the model")
    parser.add_argument("--dataset", type=str, default="trivia_qa", choices=['trivia_qa', 'geneset', 'coqa', 'nq'], help="The dataset to use")
    parser.add_argument("--split", type=str, default="validation", help="The split to evaluate on")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model to use")
    parser.add_argument("--n_test", type=int, default=10, help="The number of validation samples to test")
    parser.add_argument("--do_perturb", type=bool, default=False, help="Whether to perturb the samples")
    parser.add_argument("--n_perturb", type=int, default=1, help="The number of perturbed samples to test for each validation example")
    parser.add_argument("--n_sample", type=int, default=5, help="The number of samples to generate for each prompt")
    parser.add_argument("--embedding_model", type=str, choices=['bert', 'sapbert', 'mpnet', 'sfr-embedding-mistral'], default="sfr-embedding-mistral", help="The embedding model to use")
    parser.add_argument("--temperature", type=float, default=1, help="The temperature to use")
    parser.add_argument("--max-tokens", type=int, default=1000, help="The maximum number of tokens to generate")
    parser.add_argument("--stop", type=str, default=None, help="The stop sequence")
    parser.add_argument("--uq_metric", type=str, default="all", choices=['variance', 'entropy_exact', 'entropy_cluster', 'all'], help="The uncertainty metric to use")
    parser.add_argument("--perturb_type", type=str, default="rephrase", choices=["bootstrap", "permute"], help="The type of perturbation to use")
    parser.add_argument("--prompt_type", type=str, default="few_shot", choices=["few_shot", "zero_shot"], help="The type of prompt to use")
    parser.add_argument("--preload_responses", type=bool, default=False, help="Whether to preload the responses")
    parser.add_argument("--preload_uncertainty", type=bool, default=False, help="Whether to preload the uncertainty")
    parser.add_argument("--preload_accuracy", type=bool, default=False, help="Whether to preload the accuracy")
    parser.add_argument("--preload_calibration", type=bool, default=False, help="Whether to preload the calibration")
    parser.add_argument("--save_results", type=bool, default=False, help="Whether to save the results")

    args = parser.parse_args()
    
    experiment(args)
    # from generate_questions import generate_questions
    # generate_questions(args)

    