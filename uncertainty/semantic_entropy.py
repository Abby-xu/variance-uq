import generate_responses
from models import gpt
import argparse
import torch
from typing import List
import models
import numpy as np


def get_prompt(question, responses):
    response_str = '\n'.join([f'Response {i+1}: {response}' for i, response in enumerate(responses)])

    prompt = f'''A semantic class is a set of responses that are semantically similar. Given a question and a list of responses, respond with the number of unique semantic classes in the responses.
###
Question: What is the capital of France?

Responses:
Response 1: The capital of France is Paris.
Response 2: The French capital is Paris, France.
Response 3: France's capital is located in Berlin.
Response 4: Paris.
Response 5: The capital of Paris is Berlin.

Equivalence Classes: [{{Paris}}, {{London}}, {{Berlin}}]

Number of Semantic Classes: 3
###
Question: What is the best fruit to eat?

Responses:
Response 1: The best fruit to eat is an apple.
Response 2: An apple is the most delicious fruit.
Response 3: The tastiest fruit in the whole world is an apple.
Response 4: Apple.
Response 5: The best fruit is a banana.

Equivalence Classes: [{{apple}}, {{banana}}]

Number of Semantic Classes: 2

Question: {question}

Responses:
{response_str}

Equivalence Classes:'''

    return prompt

def parse_response(response):
    try:
        # assert there are two instances of "Number of Semantic Classes:"
        response = response.strip().lower()
        # count the number of instances of "Number of Semantic Classes:"
        num_instances = response.count("number of semantic classes:")
        assert num_instances == 1, f"Expected 2 instances of 'Number of Semantic Classes:', but found {num_instances}"

        # find the text that comes after the second instance of "Number of Semantic Classes:"
        response = response.split("number of semantic classes:")[-1]
        response = response.strip()
        # assert that the response is a number
        assert response.isdigit(), f"Expected a number, but found {response}"
        return int(response)
    except Exception as e:
        print(f'Error parsing response: {response}')
        print(e)
        return None

def calculate_num_semantic_sets(question, responses):
    print()
    print('Question:', question)
    print('Responses:', responses)
    prompt = get_prompt(question, responses)
    n_sample = 1
    temperature = 0
    model = 'gpt-3.5-turbo'
    raw_response = models.gpt(system_prompt=None, prompt=prompt, model=model, n=n_sample, temperature=temperature, max_tokens=4096)[0]
    print('Raw Response:', raw_response)
    num_semantic_sets = parse_response(raw_response)
    print('Number of Semantic Classes:', num_semantic_sets)
    return num_semantic_sets

def get_response(prompt):
    n_sample = 1
    temperature = 0
    # grade with gpt-4, could change
    # model = 'gpt-4'
    model = 'gpt-3.5-turbo'
    raw_response = models.gpt(system_prompt=None, prompt=prompt, model=model, n=n_sample, temperature=temperature)[0]
    return parse_response(raw_response)


def calculate_semantic_entropy(question, responses, wrapper):
    # TODO: add this to config
    return wrapper.get_semantic_entropy(question, responses)

class ClassifyWrapper():
    # TODO: double check this is the same model used in the original paper
    def __init__(self, model_name='microsoft/deberta-large-mnli') -> None:
        self.model_name = model_name
        self.model, self.tokenizer = models.get_entailment_model(), models.get_entailment_tokenizer()

    def entropy_over_classes(self, classes: List[List[str]]) -> float:
        '''
        Calculates the entropy over a list of classes
        '''
        n = sum([len(cls) for cls in classes])
        # get probability of each class
        probs = [len(cls) / n for cls in classes]
        # calculate entropy
        entropy = -sum([p * np.log2(p) for p in probs])
        return entropy

    def get_semantic_entropy(self, question, responses):
        '''
        Implements Algorithm 1 from Kuhn et al. https://arxiv.org/pdf/2302.09664.pdf
        '''
        print()
        print('Responses:', responses)
        classes = [[responses[0]]]
        for response_idx in range(1, len(responses)):
            # print(classes)
            response = responses[response_idx]
            lonely = True
            for cls_idx, cls in enumerate(classes):
                # take the first element from the class to make the comparison
                cls_sample = cls[0]
                # add an exact match check to speed things up
                if response == cls_sample:
                    cls.append(response)
                    lonely = False
                    break

                elif self._compare(question, cls_sample, response)['deberta_prediction'] == 1:
                    classes[cls_idx].append(response)
                    lonely = False
                    break
            if lonely:
                classes.append([response])
        print('Classes:', classes)
        print('Num Classes:', len(classes))
        entropy = self.entropy_over_classes(classes)
        print('Entropy:', entropy)
        return entropy
        
    @torch.no_grad()
    def _batch_pred(self, sen_1: list, sen_2: list, max_batch_size=128):
        inputs = [_[0] + ' [SEP] ' + _[1] for _ in zip(sen_1, sen_2)]
        inputs = self.tokenizer(inputs, padding=True, truncation=True)
        input_ids = torch.tensor(inputs['input_ids']).to(self.model.device)
        attention_mask = torch.tensor(inputs['attention_mask']).to(self.model.device)
        logits = []
        for st in range(0, len(input_ids), max_batch_size):
            ed = min(st + max_batch_size, len(input_ids))
            logits.append(self.model(input_ids=input_ids[st:ed],
                                attention_mask=attention_mask[st:ed])['logits'])
        return torch.cat(logits, dim=0)

    @torch.no_grad()
    def create_sim_mat_batched(self, question, answers):
        unique_ans = sorted(list(set(answers)))
        semantic_set_ids = {ans: i for i, ans in enumerate(unique_ans)}
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros((len(unique_ans), len(unique_ans),3))
        anss_1, anss_2, indices = [], [], []
        for i, ans_i in enumerate(unique_ans):
            for j, ans_j in enumerate(unique_ans):
                if i == j: continue
                anss_1.append(f"{question} {ans_i}")
                anss_2.append(f"{question} {ans_j}")
                indices.append((i,j))
        if len(indices) > 0:
            sim_mat_batch_flat = self._batch_pred(anss_1, anss_2)
            for _, (i,j) in enumerate(indices):
                # return preds instead of logits
                sim_mat_batch[i,j] = torch.nn.functional.softmax(sim_mat_batch_flat[_])
        return dict(
            mapping = [_rev_mapping[_] for _ in answers],
            sim_mat = sim_mat_batch
        )

    @torch.no_grad()
    def _pred(self, sen_1: str, sen_2: str):
        input = sen_1 + ' [SEP] ' + sen_2
        input_ids = self.tokenizer.encode(input, return_tensors='pt').to(self.model.device)
        out = self.model(input_ids=input_ids)
        logits = out.logits
        # logits: [Contradiction, neutral, entailment]
        return logits

    @torch.no_grad()
    def pred_qa(self, question:str, ans_1:str, ans_2:str):
        return self._pred(f"{question} {ans_1}", f'{question} {ans_2}')

    @torch.no_grad()
    def _compare(self, question:str, ans_1:str, ans_2:str):
        pred_1 = self._pred(f"{question} {ans_1}", f'{question} {ans_2}')
        pred_2 = self._pred(f"{question} {ans_2}", f'{question} {ans_1}')
        preds = torch.concat([pred_1, pred_2], 0)
        
        # Check if both predictions are entailment (index 2)
        deberta_prediction = 1 if (preds.argmax(1) == 2).all() else 0

        # deberta prediction is 1 if both predictions are entailment
        return {'deberta_prediction': deberta_prediction,
                'prob': torch.softmax(preds,1).mean(0).cpu(),
                'pred': preds.cpu()
            }