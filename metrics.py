from transformers import BertModel, BertTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer
import torch
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from rouge import Rouge
from utils import *
import argparse

def get_average_input_embedding(output: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> torch.Tensor:
    # Tokenize the input text
    tokens = tokenizer(output, return_tensors="pt")
    input_ids = tokens['input_ids']

    # Extract input embeddings directly from the model's embedding layer
    # Note: For some models, you might need to adjust this to match the specific architecture
    with torch.no_grad():
        input_embeddings = model.embeddings(input_ids)

    # Calculate the average embedding across all tokens in the output
    # As before, consider excluding special tokens like [CLS] and [SEP] based on your needs
    average_embedding = input_embeddings.mean(dim=1)  # Averaging across the sequence_length dimension

    return average_embedding

def get_glove_embedding_sentence(output: str, glove_model) -> np.ndarray:
    """
    Calculate the average GloVe embedding for a sentence.
    
    :param output: The input sentence.
    :param glove_model: The loaded GloVe model as a KeyedVectors object.
    :return: The average embedding as a numpy array.
    """
    words = output.split()
    words = [word.lower() for word in words]
    embeddings = [glove_model[word] for word in words if word in glove_model]

    if not embeddings:
        # Return zero vector if no words found in the model
        # Using the vector size from the glove_model
        return np.zeros(glove_model.vector_size)

    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding


from collections import Counter
def exact_match_entropy(outputs: List[str]) -> float:
    counter = Counter()
    counter.update(outputs)
    # get entropy
    entropy = 0
    for k, v in counter.items():
        p = v / len(outputs)
        entropy += -p * np.log2(p)
    return entropy

def get_cluster_entropy(outputs: List[str]) -> float:
    raise NotImplementedError

def rouge_l_f_score(pred, answer):
    '''
    Calculates the ROUGE-L score between the prediction and the answer
    '''
    rouge = Rouge()
    scores = rouge.get_scores(pred.lower(), answer.lower())
    return scores[0]['rouge-l']['f']

def exact_match(pred, answer):
    return pred == answer

def prompt_correctness(question, pred, answer, args):
    prompt = '''
    You are given a question, a reference answer, and predicted answer. Your task is to determine if the predicted answer is correct. You must judge corrected based on the reference answer given to you, not what you believe the answer should be.
    The predicted answer does not always need to be the exact same as the correct answer. Use your best judgment. Respond "Yes" if the predicted answer is correct, "No" otherwise.
    Question: What country is the state of California located in?
    Reference: United States
    Answer: U.S.A.
    Correct: Yes
    
    Question: What is the name of Ernest Hemingway's first novel?
    Reference: The Sun Also Rises
    Answer: The Sun Rises, Too
    Correct: No
    
    Question: What color is the sky?
    Reference: Red
    Answer: Blue
    Correct: No
    
    Question: {}
    Correct answer: {}
    Predicted answer: {}
    Correct: '''.format(question, answer, pred)

    rephrase_args = vars(args).copy()
    rephrase_args['n_sample'] = 1
    rephrase_args['temperature'] = 0
    response = generate_response(prompt, argparse.Namespace(**rephrase_args))[0]
    print(pred, answer, response)
    if response == 'Yes':
        return True
    else:
        return False


def calculate_entropy_exact(outputs: List[str]) -> float:
    counter = Counter()
    counter.update(outputs)
    # get entropy
    entropy = 0
    for k, v in counter.items():
        p = v / len(outputs)
        entropy += -p * np.log2(p)
    return entropy

def calculate_similarity_cluster_entropy(outputs: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, args: argparse.Namespace) -> float:
    if args.embedding_model == 'sapbert':
        embeddings = get_average_embedding(outputs, model, tokenizer, to_numpy=False)
    else:
        embeddings = get_cls_embedding(outputs, model, tokenizer, to_numpy=False)
    
    # group embeddings by cluster


def calculate_lexical_similarity(outputs: List[str]) -> float:
    # calculates the average rouge-l score between responses
    rouge = Rouge()
    scores = []

    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            score = rouge.get_scores(outputs[i], outputs[j], avg=True)
            scores.append(score['rouge-l']['f'])

    if len(scores) > 0:
        average_score = sum(scores) / len(scores)
    else:
        average_score = 0.0

    return average_score

def calculate_roc_auc(results, uncertainty_decomposition_results):
    # use the total uncertainty to calculate the roc auc
    total_var_uq_vals = []
    model_var_uq_vals = []
    data_var_uq_vals = []
    entropy_uq_vals = []
    lexi_sim_uq_vals = []
    num_distinct_answers_uq_vals = []
    exact_match_vals = []
    rouge_l_vals = []
    correct_per_gpt_vals = []

    for idx, example_results in enumerate(results):
        total_uncertainty = uncertainty_decomposition_results['total_uncertainty'][idx].item()
        model_uncertainty = uncertainty_decomposition_results['model_uncertainty'][idx].item()
        data_uncertainty = uncertainty_decomposition_results['data_uncertainty'][idx].item()
        outputs = []
        answer = None
        answer = example_results['answer']
        for idx_perturb, perturb_results in enumerate(example_results['results']):
            for idx_sample, sample_result in enumerate(perturb_results['results']):
                if sample_result['response'] == '':
                    output = ' '
                else:
                    output = sample_result['response']
                outputs.append(output)
                exact_match = int(sample_result['exact_match'])
                rouge_l = sample_result['rouge_l_score']
                rouge_l = int(rouge_l > 0.3)
                correct_per_gpt = sample_result['correct_per_gpt']

                # add accuracy metrics
                exact_match_vals.append(exact_match)
                rouge_l_vals.append(rouge_l)
                correct_per_gpt_vals.append(correct_per_gpt)
                
                # add var uq metrics
                total_var_uq_vals.append(total_uncertainty)
                model_var_uq_vals.append(model_uncertainty)
                data_var_uq_vals.append(data_uncertainty)

        # calculate exact match entropy
        exact_match_entropy = calculate_entropy_exact(outputs)
        entropy_uq_vals.extend([exact_match_entropy] * len(perturb_results['results']) * len(example_results['results']))

        # calculate number of distinct answers
        num_distinct_answers_uq_vals.extend([len(set(outputs))] * len(perturb_results['results']) * len(example_results['results']))

        # calculate lexi sim entropy
        lexi_sim = calculate_lexical_similarity(outputs)
        lexi_sim_uq_vals.extend([lexi_sim] * len(perturb_results['results']) * len(example_results['results']))

    total_var_uq_vals = torch.tensor(total_var_uq_vals)
    entropy_uq_vals = torch.tensor(entropy_uq_vals)
    exact_match_vals = torch.tensor(exact_match_vals)

    uq_metrics = {'total_variance': total_var_uq_vals, 'model_variance': model_var_uq_vals, 'total_variance': total_var_uq_vals, 'entropy_exact': entropy_uq_vals, 'lexi_sim': lexi_sim_uq_vals, 'num_distinct_answers': num_distinct_answers_uq_vals}
    accuracy_metrics = {'correct_per_gpt': correct_per_gpt_vals}

    roc_auc_results = []
    for uq_metric, uq_vals in uq_metrics.items():
        # if uq_vals is tensor, convert to numpy array
        if isinstance(uq_vals, torch.Tensor):
            uq_vals = uq_vals.detach().numpy()
        for accuracy_metric, accuracy_vals in accuracy_metrics.items():
            if isinstance(accuracy_vals, torch.Tensor):
                accuracy_vals = accuracy_vals.detach().numpy()
            print(f'{uq_metric}: {np.mean(uq_vals):.3}')
            print(f'{accuracy_metric}: {np.mean(accuracy_vals):.3}')
            if uq_metric == 'lexi_sim':
                roc_auc = roc_auc_score(accuracy_vals, uq_vals)
            else:
                roc_auc = 1 - roc_auc_score(accuracy_vals, uq_vals)
            roc_auc_results.append({'uq_metric': uq_metric, 'accuracy_metric': accuracy_metric, 'roc_auc': roc_auc})
            print(f'AUROC: {roc_auc:.3}')
            print()

    return roc_auc_results
    
def calculate_uncertainty_at_pos(sample: torch.Tensor, uncertainty_type: str) -> torch.Tensor:
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

def calculate_uncertainty(embeddings: torch.Tensor, uncertainty_type: str) -> torch.Tensor:
    results = []
    for i in range(embeddings.size(-1)):
        uncertainty = calculate_uncertainty_at_pos(embeddings[..., i], uncertainty_type)
        results.append(uncertainty)
    return torch.stack(results, dim=-1).sum(dim=-1)

# Now define the specific functions for model, data, and total uncertainty
def calculate_model_uncertainty(embeddings: torch.Tensor) -> torch.Tensor:
    return calculate_uncertainty(embeddings, 'model')

def calculate_data_uncertainty(embeddings: torch.Tensor) -> torch.Tensor:
    return calculate_uncertainty(embeddings, 'data')

def calculate_total_uncertainty(embeddings: torch.Tensor) -> torch.Tensor:
    return calculate_uncertainty(embeddings, 'total')







