from transformers import AutoModel, AutoTokenizer

def get_mpnet_model():
    return AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def get_mpnet_tokenizer():
    return AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")