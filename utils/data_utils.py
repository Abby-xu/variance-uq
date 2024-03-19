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
    
    prompt = '''Rephrase the following trivia question in your own words. The rephrased question should ask the same thing as the original question, just expressed differently. The answer to the rephrased question should be the same as the answer to the original question.
Original question: What is the capital city of Australia?
Rephrased question: Which Australian city houses the country's parliament and serves as its capital?
Original question: {}
Rephrased question:'''.format(sample)
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
    encoded_inputs = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt')

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