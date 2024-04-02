import generate_responses
from models import gpt
import argparse


def parse_response(response):
    return response.strip().lower()

def get_response(prompt):
    n_sample = 1
    temperature = 0
    # grade with gpt-4, could change
    # model = 'gpt-4'
    model = 'gpt-3.5-turbo'
    raw_response = gpt(system_prompt=None, prompt=prompt, model=model, n=n_sample, temperature=temperature)[0]
    return parse_response(raw_response)

def calculate_ask_for_accuracy(question, response, answer, args):
    # TODO: maybe better if score valued between 0 and 100
    prompt = '''
    Imagine you are a teacher grading a quiz. You are given a question, a reference answer, and predicted answer. Your task is to determine if the predicted answer is correct. You must judge corrected based on the reference answer given to you, not what you believe the answer should be.
    The predicted answer does not need to be the exact same as the correct answer. The predicted answer can be phrased differently than the reference answer. What matters is that the semantics are the same. Use your best judgment. Respond "Yes" if the predicted answer is correct, "No" otherwise.

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
    Correct:'''.format(question, answer, response)

    rephrase_args = vars(args).copy()
    rephrase_args['n_sample'] = 1
    rephrase_args['temperature'] = 0
    evaluation_response = get_response(prompt)
    print('Question:', question)
    print('Correct answer:', answer)
    print('Predicted answer:', response)
    print('Correct:', evaluation_response)
    print()

    if evaluation_response == 'yes':
        return True
    elif evaluation_response == 'no':
        return False
    else:
        print('Invalid response:', evaluation_response)
        return None