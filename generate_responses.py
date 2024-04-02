import openai
from utils import data_utils
import numpy as np
import einops

def generate_responses(args) -> np.ndarray:
    '''
    Generates responses for the given dataset
    Responses are stored in a tensor of shape (n_samples, n_perturb, n_responses)
    '''

    original_questions = []
    questions = []
    responses = []
    answers = []

    dataset = data_utils.load_dataset(args)

    print("=" * 50)
    print("Generating Responses".center(50))
    print("=" * 50)

    for idx_test, data in enumerate(dataset):
        if args.dataset == 'trivia_qa':
            input = data['input']
            answer = data['answer']['value']
        elif args.dataset == 'geneset':
            input = data['input'].split(' ')
            answer = data['answer']
        else:
            raise NotImplementedError

        original_questions.append((einops.repeat(np.array([input]), 'b -> b n_perturb n_sample', b=1, n_perturb=args.n_perturb, n_sample=args.n_sample)).tolist()[0])

        print(f"\nQuestion {idx_test + 1}")
        print("-" * 20)
        print(f"Original Question: {input}")
        print(f"Answer: {answer}")

        example_questions = []
        example_responses = []
        example_answers = []

        # if we turn perturbation off, we only generate one question for each example
        # and that question is the original question
        if args.do_perturb == False:
            n_perturb = 1
        else:
            n_perturb = args.n_perturb

        for idx_perturb in range(n_perturb):
            perturb_questions = []
            perturb_responses = []
            perturb_answer_results = []

            if args.do_perturb is True:
                sample = data_utils.perturb_sample(input, args)
            else:
                sample = input
            
            if args.dataset == 'geneset':
                prompt = data_utils.geneset_sample_to_prompt(sample)
            else:
                if args.prompt_type == 'few_shot':
                    prompt = data_utils.sample_to_prompt(sample)
                else:
                    prompt = data_utils.sample_to_prompt_zero_shot(sample) # redundant, just returns prompt

            sample_responses = data_utils.generate_response(prompt, args)

            perturb_questions.append(sample)
            perturb_responses.extend(sample_responses)
            perturb_answer_results.extend([answer] * len(sample_responses))

            print(f"\nPerturbation {idx_perturb + 1}")
            print("~" * 10)
            print(f"Perturbed Question: {sample}")
            print("Responses:")
            for idx, response in enumerate(sample_responses):
                print(f"{idx + 1}. {response}")

            example_questions.append(perturb_questions)
            example_responses.append(perturb_responses)
            example_answers.append(perturb_answer_results)

        questions.append(example_questions)
        responses.append(example_responses)
        answers.append(example_answers)

    original_questions_array = np.array(original_questions, dtype=str)
    questions_array = np.array(questions, dtype=str)
    responses_array = np.array(responses, dtype=str)
    answers_array = np.array(answers, dtype=str)

    print("\n" + "=" * 50)
    print("Response Generation Complete".center(50))
    print("=" * 50)

    return original_questions_array, questions_array, responses_array, answers_array

if __name__ == "__main__":
    {'index': 199, 'input': 'Who did Jack Ruby shoot in November 1963?', 'answer': {'aliases': ['Oswald the Lone Assassin', 'Lone Nut Theory', 'Lone gunman', 'Lee Oswald', 'Lee H. Oswald', 'A.J. Hidell', 'L.H.O.', 'L. H. Oswald', 'L.H.O', 'Alek J. Hidell', 'Maria Oswald Porter', 'Lee harvey oswald', 'Lee Harvey Oswald (photo)', "Lee Harvey Oswald's", 'Lee Harvey Oswald', 'Lee harvy oswald', 'Lone gunman theory', 'Alek James Hidell', 'Lee Harvey Ostwald'], 'normalized_aliases': ['lee harvey oswald', 'lone nut theory', 'oswald lone assassin', 'l h oswald', 'lee oswald', 'lee harvey ostwald', 'lee h oswald', 'alek james hidell', 'lee harvy oswald', 'j hidell', 'lee harvey oswald photo', 'lone gunman', 'l h o', 'lee harvey oswald s', 'maria oswald porter', 'alek j hidell', 'lone gunman theory'], 'matched_wiki_entity_name': '', 'normalized_matched_wiki_entity_name': '', 'normalized_value': 'lee harvey oswald', 'type': 'WikipediaEntity', 'value': 'Lee Harvey Oswald'}}
    pass