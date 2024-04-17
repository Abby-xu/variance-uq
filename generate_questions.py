import pandas as pd
from utils import data_utils
import json

def generate_questions(args):

    with open('results_clean_50_4_10.csv') as f:

        # BEGIN TEMP
        df = pd.read_csv(f)
        questions = df['original_question'].values
        perturbed_questions = df['perturbed_question'].values

        # create dict mapping original question to perturbed questions
        cache_perturbed_questions_dict = {}
        for idx, question in enumerate(questions):
            if question not in cache_perturbed_questions_dict:
                cache_perturbed_questions_dict[question] = []
            cache_perturbed_questions_dict[question].append(perturbed_questions[idx])
        
        for original_question in cache_perturbed_questions_dict:
            cache_perturbed_questions_dict[original_question] = list(set(cache_perturbed_questions_dict[original_question]))
            print()
            print('Found Question:', original_question)
            print('Num Perturbs in Cache:', len(cache_perturbed_questions_dict[original_question]))
        # END TEMP

    perturbed_questions_dict = {}
    dataset = data_utils.load_dataset(args)
    for idx_test, data in enumerate(dataset):
        if args.dataset == 'trivia_qa':
            idx = data['index']
            inp = data['input']
            answer = data['answer']['value']
        else:
            raise NotImplementedError
        
        sample_perturbed_questions = [inp]

        # BEGIN TEMP
        if inp in cache_perturbed_questions_dict:
            sample_perturbed_questions.extend(cache_perturbed_questions_dict[inp])
        new_n_perturb = args.n_perturb - len(sample_perturbed_questions)
        print()
        print(f'Num Perturbs in Cache for Index {idx}:', len(sample_perturbed_questions))
        print(f'Num Perturbs to Generate for Index {idx}:', new_n_perturb)

        print(list(set(sample_perturbed_questions)))
        for _perturb_idx in range(new_n_perturb): # we do -1 because we include the original question
        # END TEMP
            perturbed_question = None
            while perturbed_question in sample_perturbed_questions or perturbed_question is None:
                perturbed_question = data_utils.rephrase_question(inp, args)
                print(perturbed_question)
                print(perturbed_question in sample_perturbed_questions)
            sample_perturbed_questions.append(perturbed_question)
        perturbed_questions_dict[idx] = sample_perturbed_questions
        print(list(set(sample_perturbed_questions)))


    # save to json
    questions_dir = 'questions'
    with open(f'{questions_dir}/{args.dataset}.json', 'w') as f:
        json.dump(perturbed_questions_dict, f)
    
    return perturbed_questions_dict
