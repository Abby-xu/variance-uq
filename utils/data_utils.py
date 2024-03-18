import os
import openai
import models
import metrics
import numpy as np
import datasets
import pandas as pd
from typing import List
import torch
from jaxtyping import Float, Int, Bool, Str
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import argparse

openai.api_key = os.environ['OPENAI_API_KEY']

def sample_to_prompt(question, **kwargs):
    if isinstance(question, list):
        return [sample_to_prompt(q, **kwargs) for q in question]
    return f"""Answer these questions:
Q: In Scotland a bothy/bothie is a?
A: House
Q: {question}
A:"""

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
    
    prompt = '''Rephrase the following trivia question in your own words. The rephrased question should be asking the same thing as the original question, just expressed differently.
Original question: What is the capital city of Australia?
Rephrased question: Which Australian city houses the country's parliament and serves as its capital?
Original question: {}
Rephrased question: '''.format(sample)
    rephrase_args = vars(args).copy()
    rephrase_args['n_sample'] = 1
    # rephrase_args['model'] = 'gpt-4'
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

def load_dataset(args):
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
        # only choose args.n_test samples
        data = data[:args.n_test]
        return data
    
    elif args.dataset == 'geneset':
        # TODO: exclude examples over a certain number of genes?
        filename = 'geneset.csv'
        # create file if it does not exist already
        if not os.path.exists(filename):
            x_filename = '/Users/kylecox/Documents/ws/tot-gene-sets/src/tot/data/gene_sets/x_eval.txt'
            y_filename = '/Users/kylecox/Documents/ws/tot-gene-sets/src/tot/data/gene_sets/y_eval.txt'
            with open(x_filename, 'r') as f:
                x = f.readlines()
            x = [d.strip() for d in x]
            x = {i: d for i, d in enumerate(x)}
            with open(y_filename, 'r') as f:
                y = f.readlines()
            y = [d.strip() for d in y]
            y = {i: d for i, d in enumerate(y)}
            data = []
            for i in range(len(x)):
                data.append({'index': i, 'input': x[i], 'answer': y[i]})
            data = pd.DataFrame(data)
            data.to_csv(filename)
        data = pd.read_csv(filename)
        data['index'] = data.index
        data = data.to_dict(orient='records')
        # choose all the 0 through args.n_test-1 samples
        return data[:args.n_test]

    else:
        raise NotImplementedError

def parse_response(response):
    response = response.split('\n')[0]
    response = response.strip()
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]
    elif response.startswith("'") and response.endswith("'"):
        response = response[1:-1]
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

def get_cls_embeddings(outputs: torch.Tensor, model, tokenizer, to_numpy=True) -> torch.Tensor:
    """
    Calculate the [CLS] token embeddings from the last layer of the model for multiple outputs.
    
    Args:
        outputs: Tensor of strings with shape (batch_size,)
        model: The pre-trained model
        tokenizer: The tokenizer associated with the model
        to_numpy: Whether to convert the embeddings to a NumPy array (default: True)
        
    Returns:
        sentence_embeddings: Tensor of shape (batch_size, embedding_dim)
    """
    encoded_inputs = tokenizer(outputs.tolist(), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_inputs)
    
    # Get the [CLS] token embeddings from the last layer
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]
    
    if to_numpy:
        sentence_embeddings = sentence_embeddings.cpu().numpy()
    
    return sentence_embeddings

def get_all_embeddings(
    results: Str[Tensor, '...'],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    normalize: bool = True,
    batch_size: int = 32,
) -> Float[Tensor, '..., embedding_dim']:
    input_shape = results.shape
    output_shape = input_shape + (model.config.hidden_size,)
    out = torch.zeros(output_shape)

    results_flat = results.view(-1)
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

        out_flat = out.view(-1, model.config.hidden_size)
        out_flat[start_idx:end_idx] = batch_embeddings

    return out