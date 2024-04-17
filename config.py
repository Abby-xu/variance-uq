import models
import argparse
import torch
import os

# TODO: make global config variable to replace args

# should probably move dataset here too

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
openai_api_key = os.environ['OPENAI_API_KEY']

def initialize_components(args):
    if args is None:
        tokenizer = models.get_mpnet_tokenizer()
        model = models.get_mpnet_model()
        return tokenizer, model
    
    if args.embedding_model == 'bert':
        tokenizer = models.get_bert_tokenizer()
        model = models.get_bert_model()
        return tokenizer, model
    elif args.embedding_model == 'mpnet':
        tokenizer = models.get_mpnet_tokenizer()
        model = models.get_mpnet_model()
        return tokenizer, model
    elif args.embedding_model == 'sapbert':
        tokenizer = models.get_sapbert_model()
        model = models.get_sapbert_tokenizer()
        return tokenizer, model
    elif args.embedding_model == 'entailment':
        tokenizer = models.get_entailment_tokenizer()
        model = models.get_entailment_model()
        return tokenizer, model
    else:
        raise ValueError("Invalid embedding model specified.")