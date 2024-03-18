import openai
from utils import data_utils

import torch

def generate_responses(args) -> torch.Tensor:
    '''
    Generates responses for the given dataset
    Responses are stored in a tensor of shape (n_samples, n_perturb, n_responses)
    '''
    results = []
    dataset = data_utils.load_dataset(args)
    
    for idx_test, data in enumerate(dataset):
        if args.dataset == 'trivia_qa':
            input = data['input']
            answer = data['answer']['aliases'][0]
        elif args.dataset == 'geneset':
            input = data['input'].split(' ')
            answer = data['answer']
        else:
            raise NotImplementedError
        
        example_results = []
        
        for idx_perturb in range(args.n_perturb):
            perturb_results = []
            perturbed_sample = data_utils.perturb_sample(input, args)
            
            if args.dataset == 'geneset':
                prompt = data_utils.geneset_sample_to_prompt(perturbed_sample)
            else:
                prompt = data_utils.sample_to_prompt(perturbed_sample)
            
            responses = data_utils.generate_response(prompt, args)
            
            for response in responses:
                perturb_results.append(response)
            
            example_results.append(perturb_results)
        
        results.append(example_results)
    
    results_tensor = torch.tensor(results, dtype=torch.string)
    
    return results_tensor

if __name__ == "__main__":
    pass