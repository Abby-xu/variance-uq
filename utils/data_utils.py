import os
import openai
import models
import numpy as np
import datasets
import pandas as pd
from typing import List
import torch
from jaxtyping import Float, Int, Bool
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import argparse
import json

device = config.device
openai.api_key = config.openai_api_key

def load_questions(args):
    questions_dir = 'questions'
    questions_filename = f'{args.dataset}.json'
    questions_filename = os.path.join(questions_dir, questions_filename)
    questions = load_json(questions_filename)
    return questions

def load_json(filename):
    with open(filename, 'r') as f:
        dic = json.load(f)
        dic = {int(k): v for k, v in dic.items()}
    return dic
    
def sample_from_perturbed_questions(idx, perturbed_questions, args):
    n_perturb = args.n_perturb
    perturbed_questions = perturbed_questions[idx]
    assert len(list(set(perturbed_questions))) >= n_perturb, f"Number of perturbed questions ({len(perturbed_questions)}) is less than the number of perturbations requested ({n_perturb})."
    perturbed_questions = np.random.choice(perturbed_questions, n_perturb, replace=False)
    return perturbed_questions

def sample_to_prompt(question, full_sentence_response=False, **kwargs):
    if isinstance(question, list):
        return [sample_to_prompt(q, **kwargs) for q in question]
    if full_sentence_response is False:
        return f"""Answer the following questions. Your answers should be short, only a word or phrase. 
    Q: Who won Super Bowl XX?
    A: The Chicago Bears
    Q: {question}
    A:"""
    else:
        return f"""Answer the following questions. Respond to each question with a full sentence, including the context of the question in your answer.
    Q: Who won Super Bowl XX?
    A: The Chicago Bears won Super Bowl XX.
    Q: {question}
    A:"""

def sample_to_prompt_zero_shot(question, **kwargs):
    if isinstance(question, list):
        return [sample_to_prompt_zero_shot(q, **kwargs) for q in question]
    return question

def perturb_sample(sample, args):
    if args.perturb_type == 'bootstrap':
        return subsample_geneset(sample)
    elif args.perturb_type == 'permute':
        return permute_geneset(sample)
    elif args.perturb_type == 'rephrase':
        return rephrase_question(sample, args)
    else:
        raise NotImplementedError
    
def rephrase_question(sample, args):
    # send prompt to model
    # sample is the original prompt

    prompt = '''Rephrase the following trivia question in your own words. The rephrased question should preserve the meaning of the original question, but be worded differently. The answer to the rephrased question should be the same as the answer to the original question. You can be creative.
Original question: What is the capital city of Australia?
Rephrased question: Which Australian city serves as the country's capital?
Original question: {}
Rephrased question:'''.format(sample)
    rephrase_args = vars(args).copy()
    rephrase_args['n_sample'] = 1
    # rephrase_args['model'] = 'gpt-3.5-turbo'
    rephrase_args['model'] = 'gpt-4'
    rephrase_args['temperature'] = 0.9
    response = generate_response(prompt, argparse.Namespace(**rephrase_args))[0]
    return response


def permute_geneset(sample):
    np.random.shuffle(sample)
    return sample

def subsample_geneset(sample: List[str], frac: float=0.5) -> List[str]:
    n = int(len(sample) * frac)
    np.random.shuffle(sample)
    return sample[:n]

def geneset_sample_to_prompt(sample: List[str]):
    serialized_sample = ', '.join(sample)
    return f"Give a name for the most prominent biological process performed by the following set of genes: {serialized_sample}."

def load_dataset(args, shuffle=False):
    if args.dataset == 'trivia_qa':
        split = 'validation'
        data = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
        id_mem = set()

        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        data = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
        assert pd.Series([_['question_id'] for _ in data]).value_counts().max() == 1

        # Convert the dataset to a list of dictionaries with only "index", "input", and "answer" keys
        data = [{'index': i, 'input': _['question'], 'answer': _['answer']}
                for i, _ in enumerate(data)]
        
        if shuffle:
            np.random.shuffle(data)

        # only choose args.n_test samples
        data = data[:args.n_test]
        return data
    
    elif args.dataset == 'coqa':
        # conversational qa
        split = 'validation'
        data = datasets.load_dataset("coqa", split=split)
        
        data = [{
            'index': i,
            'input': f"{_['story']} Q: {_['questions'][0]['input_text']}",
            'answer': _['answers'][0]['input_text']
        } for i, _ in enumerate(data)]
        
        if shuffle:
            np.random.shuffle(data)
            
        # only choose args.n_test samples
        data = data[:args.n_test]
        return data
        
        # raise NotImplementedError

    elif args.dataset == 'nq':
        # natural questions
        split = 'validation'
        data = datasets.load_dataset("nq_open", split=split)
        
        data = [_ for _ in data if _['annotations']['short_answers']]
        
        data = [{
            'index': i,
            'input': _['question'],
            'answer': _['annotations']['short_answers'][0]['text']
        } for i, _ in enumerate(data) if _['annotations']['short_answers']]
        
        if shuffle:
            np.random.shuffle(data)
            
        data = data[:args.n_test]
        return data
        
        # raise NotImplementedError
    
    else:
        raise NotImplementedError

def parse_response(response):
    response = response.split('\n')[0]
    response = response.strip()
    return response

def generate_response(prompt, args):
    # TODO: test other models
    completions = models.gpt(
        system_prompt=None,
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_sample,
        stop=args.stop
    )
    return [parse_response(response) for response in completions]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_average_embedding(outputs: List[str], model, tokenizer, to_numpy=True) -> torch.Tensor:
    # TODO: fix

    encoded_inputs = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt')
    encoded_inputs = encoded_inputs.to(device)
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_inputs)
    
    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_inputs['attention_mask'])
    
    # to numpy
    if to_numpy:
        sentence_embeddings = sentence_embeddings.cpu().numpy()
    
    return sentence_embeddings

def get_average_embeddings():
    pass

def get_cls_embeddings(outputs: np.array, model, tokenizer, to_numpy=True) -> torch.Tensor:
    """
    Calculate the [CLS] token embeddings from the last layer of the model for multiple outputs.

    Args:
        outputs: Array of strings with shape (batch_size,)
        model: The pre-trained model
        tokenizer: The tokenizer associated with the model
        to_numpy: Whether to convert the embeddings to a NumPy array (default: True)

    Returns:
        sentence_embeddings: Tensor of shape (batch_size, embedding_dim)
    """
    outputs = outputs.tolist()

    # Move the model to the CPU
    model = model.to('cpu')

    encoded_inputs = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt')

    # Move the encoded inputs to the CPU
    encoded_inputs = {k: v.to('cpu') for k, v in encoded_inputs.items()}

    with torch.no_grad():
        model_output = model(**encoded_inputs)

    # Get the [CLS] token embeddings from the last layer
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]

    if to_numpy:
        sentence_embeddings = sentence_embeddings.cpu().numpy()

    return sentence_embeddings

def get_all_embeddings(
    results,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    normalize: bool = True,
    batch_size: int = 32,
) -> Float[Tensor, '... embedding_dim']:
    input_shape = results.shape
    output_shape = model.config.hidden_size
    out = torch.zeros(len(results.flatten()), output_shape)
    
    results_flat = results.flatten()
    num_batches = (len(results_flat) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(results_flat))
        batch_responses = results_flat[start_idx:end_idx]
        
        if args.embedding_model == 'sapbert':
            batch_embeddings = get_average_embeddings(batch_responses, model, tokenizer, to_numpy=False)
        else:
            batch_embeddings = get_cls_embeddings(batch_responses, model, tokenizer, to_numpy=False)
        
        if normalize:
            batch_embeddings = batch_embeddings / torch.norm(batch_embeddings, dim=-1, keepdim=True)
        
        out[start_idx:end_idx] = batch_embeddings
    
    out = out.view(*input_shape, -1)
    return out

from sklearn.decomposition import PCA

def get_pca_embeddings(results, model, tokenizer, args):
    # TODO: check embedding dim
    embedding_dim = 768
    n_pca_components = 20
    out = torch.zeros(len(results), len(results[0]['results']), len(results[0]['results'][0]['results']), n_pca_components)

    for test_idx, test_sample in enumerate(results):
        # aggregate embeddings in the test sample (i.e. across all perturbed samples)
        # we do not yet know the dimension of the embedings
        test_sample_embeddings = torch.zeros(len(test_sample['results']), len(test_sample['results'][0]['results']), embedding_dim)
        for perturb_idx, perturbed_sample in enumerate(test_sample['results']):
            perturbed_responses = [trial['response'] for trial in perturbed_sample['results']]
            perturbed_embeddings = get_cls_embedding(perturbed_responses, model, tokenizer, to_numpy=False)
            perturbed_embeddings = perturbed_embeddings / torch.norm(perturbed_embeddings, dim=1, keepdim=True)
            test_sample_embeddings[perturb_idx, :, :] = perturbed_embeddings

        
        # Reshape test_sample_embeddings to perform PCA on the embedding dimension
        test_sample_embeddings_reshaped = test_sample_embeddings.reshape(-1, embedding_dim)

        # Convert the reshaped tensor to a NumPy array
        test_sample_embeddings_np = test_sample_embeddings_reshaped.numpy()

        # take pca on the reshaped test sample embeddings
        pca = PCA(n_components=n_pca_components)
        pca_embeddings = pca.fit_transform(test_sample_embeddings_np)

        # Reshape the PCA embeddings back to the original shape
        out[test_idx] = torch.from_numpy(pca_embeddings).reshape(len(test_sample['results']), len(test_sample['results'][0]['results']), n_pca_components)
    
    return out

def get_most_frequent_completion(results):
    for sample in results:
        sample['most_frequent_completion'] = max(set(sample['response']), key=sample['response'].count)