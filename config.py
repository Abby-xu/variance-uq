import models
import argparse

# TODO: make global config variable to replace args

def initialize_components(args):
    if args is None:
        # model = models.get_mpnet_model()
        # tokenizer = models.get_mpnet_tokenizer()
        # return model, tokenizer
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