import os
import openai
import backoff 
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from gensim.scripts.glove2word2vec import glove2word2vec


completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(system_prompt, prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None, json=False) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        if json:
            res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop,  response_format={'type': 'json_object'})
        else:
            res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="gpt-4"):
    # TODO: I think cost has been changed
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

def get_model(model_name="bert-base-uncased"):
    if model_name == 'bert-base-uncased':
        return BertModel.from_pretrained(model_name)
    elif model_name == 'sapbert':
        return AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    elif model_name == 'all-mpnet-base-v2':
        return AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    else:
        raise NotImplementedError

def get_tokenizer(model_name="bert-base-uncased"):
    if model_name == 'bert-base-uncased':
        return BertTokenizer.from_pretrained(model_name)
    elif model_name == 'sapbert':
        return AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") 
    elif model_name == 'all-mpnet-base-v2':
        return AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    else:
        raise NotImplementedError

def load_glove_model(path):
    import gensim
    glove_txt_file = '/Users/kylecox/Documents/ws/glove/glove.6B.300d.txt'
    glove_w2v_file = '/Users/kylecox/Documents/ws/glove/glove.6B.300d.w2v.txt'
    if not os.path.exists('/Users/kylecox/Documents/ws/glove/glove.6B.300d.w2v.txt'):
        glove_txt_file = '/Users/kylecox/Documents/ws/glove/glove.6B.300d.txt'
        glove_w2v_file = '/Users/kylecox/Documents/ws/glove/glove.6B.300d.w2v.txt'
        glove2word2vec(glove_txt_file, glove_w2v_file)
    model = gensim.models.KeyedVectors.load_word2vec_format(glove_w2v_file, binary=False)

    return model

