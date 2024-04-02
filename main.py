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

openai.api_key = os.environ['OPENAI_API_KEY']
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def experiment(args, preload_responses=False, preload_uncertainty=False, preload_accuracy=False, preload_calibration=False, save_results=False):
    tokenizer, model = config.initialize_components(args)
    # tokenizer = tokenizer.to(device)
    model = model.to(device)
    if preload_responses:
        original_questions_save_filename_no_timestamp = f'original_questions_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        # original_questions_save_filepath = os.path.join('responses', original_questions_save_filename_no_timestamp)
        with open(original_questions_save_filename_no_timestamp, 'rb') as f:
            original_questions = pickle.load(f)
        
        perturbed_questions_save_filename_no_timestamp = f'perturbed_questions_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        # perturbed_questions_save_filepath = os.path.join('responses', perturbed_questions_save_filename_no_timestamp)
        with open(perturbed_questions_save_filename_no_timestamp, 'rb') as f:
            perturbed_questions = pickle.load(f)

        responses_save_filename_no_timestamp = f'responses_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        # responses_save_filepath = os.path.join('responses', responses_save_filename_no_timestamp)
        with open(responses_save_filename_no_timestamp, 'rb') as f:
            responses = pickle.load(f)

        answers_save_filename_no_timestamp = f'answers_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        # answers_save_filepath = os.path.join('responses', answers_save_filename_no_timestamp)
        with open(answers_save_filename_no_timestamp, 'rb') as f:
            answers = pickle.load(f)

    else:
        original_questions, perturbed_questions, responses, answers = generate_responses(args)

    if preload_uncertainty:
        uncertainty_save_filename_no_timestamp = f'uncertainty_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        uncertainty_save_filepath = os.path.join('results', uncertainty_save_filename_no_timestamp)
        with open(uncertainty_save_filepath, 'rb') as f:
            uncertainty_results = pickle.load(f)
    else:
        uncertainty_results = evaluate_uncertainty(responses, tokenizer, model, args, original_questions=original_questions) # dict of tensors of shape (n_samples)

    if preload_accuracy:
        accuracy_save_filename_no_timestamp = f'accuracy_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        # accuracy_save_filepath = os.path.join('results', accuracy_save_filename_no_timestamp)
        with open(accuracy_save_filename_no_timestamp, 'rb') as f:
            accuracy_results = pickle.load(f)
    else:
        # When we do ask_for_accuracy, we pass in the original question, not the perturbed question. This could be changed.
        accuracy_results = evaluate_accuracy(original_questions, responses, answers, args) # dict of tensors of shape (n_samples, n_perturb, n_responses)
    
    if preload_calibration:
        calibration_save_filename_no_timestamp = f'calibration_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        calibration_save_filepath = os.path.join('results', calibration_save_filename_no_timestamp)
        with open(calibration_save_filepath, 'rb') as f:
            calibration_results = pickle.load(f)
    else:
        calibration_results = evaluate_calibration(uncertainty_results, accuracy_results) # dict of roc_auc values, each key refers to the pair of uncertainty and accuracy metrics


    if save_results:
        timestamp = logging_utils.get_timestamp()
        original_questions_save_filname = f'original_questions_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        original_questions_save_filename_no_timestamp = f'original_questions_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl' 
        original_questions_save_filepath = os.path.join('responses', original_questions_save_filname)
        os.makedirs(os.path.dirname(original_questions_save_filepath), exist_ok=True)
        with open(original_questions_save_filepath, 'wb') as f:
            pickle.dump(original_questions, f)
        with open(original_questions_save_filename_no_timestamp, 'wb') as f:
            pickle.dump(original_questions, f)
        
        perturbed_questions_save_filename = f'perturbed_questions_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        perturbed_questions_save_filename_no_timestamp = f'perturbed_questions_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        perturbed_questions_save_filepath = os.path.join('responses', perturbed_questions_save_filename)
        os.makedirs(os.path.dirname(perturbed_questions_save_filepath), exist_ok=True)
        with open(perturbed_questions_save_filepath, 'wb') as f:
            pickle.dump(perturbed_questions, f)
        with open(perturbed_questions_save_filename_no_timestamp, 'wb') as f:
            pickle.dump(perturbed_questions, f)

        responses_save_filename = f'responses_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        responses_save_filename_no_timestamp = f'responses_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        results_save_filepath = os.path.join('responses', responses_save_filename)
        os.makedirs(os.path.dirname(results_save_filepath), exist_ok=True)
        with open(results_save_filepath, 'wb') as f:
            pickle.dump(responses, f)
        with open(responses_save_filename_no_timestamp, 'wb') as f:
            pickle.dump(responses, f)

        answers_save_filename = f'answers_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        answers_save_filename_no_timestamp = f'answers_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        answers_save_filepath = os.path.join('responses', answers_save_filename)
        os.makedirs(os.path.dirname(answers_save_filepath), exist_ok=True)
        with open(answers_save_filepath, 'wb') as f:
            pickle.dump(answers, f)
        with open(answers_save_filename_no_timestamp, 'wb') as f:
            pickle.dump(answers, f)


        uncertainty_save_filename = f'uncertainty_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        uncertainty_save_filename_no_timestamp = f'uncertainty_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        uncertainty_save_filepath = os.path.join('results', uncertainty_save_filename)
        os.makedirs(os.path.dirname(uncertainty_save_filepath), exist_ok=True)
        with open(uncertainty_save_filepath, 'wb') as f:
            pickle.dump(uncertainty_results, f)
        with open(uncertainty_save_filename_no_timestamp, 'wb') as f:
            pickle.dump(uncertainty_results, f)

        accuracy_save_filename = f'accuracy_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        accuracy_save_filename_no_timestamp = f'accuracy_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        accuracy_save_filepath = os.path.join('results', accuracy_save_filename)
        os.makedirs(os.path.dirname(accuracy_save_filepath), exist_ok=True)
        with open(accuracy_save_filepath, 'wb') as f:
            pickle.dump(accuracy_results, f)
        with open(accuracy_save_filename_no_timestamp, 'wb') as f:
            pickle.dump(accuracy_results, f)

        calibration_save_filename = f'calibration_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}_{timestamp}.pkl'
        calibration_save_filename_no_timestamp = f'calibration_{args.dataset}_{args.split}_{args.model}_{args.uq_metric}.pkl'
        calibration_save_filepath = os.path.join('results', calibration_save_filename)
        os.makedirs(os.path.dirname(calibration_save_filepath), exist_ok=True)
        with open(calibration_save_filepath, 'wb') as f:
            pickle.dump(calibration_results, f)
        with open(calibration_save_filename_no_timestamp, 'wb') as f:
            pickle.dump(calibration_results, f)

    return responses, answers, uncertainty_results, accuracy_results, calibration_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation of the model")
    parser.add_argument("--dataset", type=str, default="trivia_qa", choices=['trivia_qa', 'geneset'], help="The dataset to use")
    parser.add_argument("--split", type=str, default="validation", help="The split to evaluate on")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model to use")
    parser.add_argument("--n_test", type=int, default=10, help="The number of validation samples to test")
    parser.add_argument("--do_perturb", type=bool, default=True, help="Whether to perturb the samples")
    parser.add_argument("--n_perturb", type=int, default=5, help="The number of perturbed samples to test for each validation example")
    parser.add_argument("--n_sample", type=int, default=2, help="The number of samples to generate for each prompt")
    parser.add_argument("--embedding_model", type=str, choices=['bert', 'sapbert', 'mpnet'], default="mpnet", help="The embedding model to use")
    parser.add_argument("--temperature", type=float, default=1, help="The temperature to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="The maximum number of tokens to generate")
    parser.add_argument("--stop", type=str, default=None, help="The stop sequence")
    parser.add_argument("--uq_metric", type=str, default="all", choices=['variance', 'entropy_exact', 'entropy_cluster', 'all'], help="The uncertainty metric to use")
    parser.add_argument("--perturb_type", type=str, default="rephrase", choices=["bootstrap", "permute"], help="The type of perturbation to use")
    parser.add_argument("--prompt_type", type=str, default="few_shot", choices=["few_shot", "zero_shot"], help="The type of prompt to use")

    args = parser.parse_args()
    
    experiment(args)

    