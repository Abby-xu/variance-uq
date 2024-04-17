from utils import data_utils
import numpy as np
import einops

def generate_responses(args) -> np.ndarray:
    '''
    Generates responses for the given dataset
    Responses are stored in a tensor of shape (n_samples, n_perturb, n_responses)
    '''

    original_questions = []
    perturbed_questions = []
    responses = []
    answers = []

    dataset = data_utils.load_dataset(args)
    if args.do_perturb: # if we are doing the perturbation, assume we have already run the perturbation script
        perturbed_questions_dict = data_utils.load_questions(args)

    print("=" * 50)
    print("Generating Responses".center(50))
    print("=" * 50)

    for idx_test, data in enumerate(dataset):
        # hopefully keep this same structure for all datasets
        if args.dataset == 'trivia_qa':
            idx = data['index']
            inp = data['input']
            answer = data['answer']['value']
        # TODO: coqa, nq
        else:
            raise NotImplementedError

        original_questions.append((einops.repeat(np.array([inp]), 'b -> b n_perturb n_sample', b=1, n_perturb=args.n_perturb, n_sample=args.n_sample)).tolist()[0])

        print(f"\nQuestion {idx_test + 1}")
        print("-" * 20)
        print(f"Original Question: {inp}")
        print(f"Answer: {answer}")

        example_questions = []
        example_responses = []
        example_answers = []

        # if we turn perturbation off, we only generate one question for each example
        # and that question is the original question
        if args.do_perturb is False:
            n_perturb = 1
            sample_perturb_questions = [inp]
        else:
            n_perturb = args.n_perturb
            sample_perturb_questions = data_utils.sample_from_perturbed_questions(idx, perturbed_questions_dict, args)

        for idx_perturb in range(n_perturb):
            sample_perturbed_question = sample_perturb_questions[idx_perturb]
            sample_perturbed_questions = []
            sample_perturbed_responses = []
            sample_perturbed_answer_results = []

            if args.prompt_type == 'few_shot':
                prompt = data_utils.sample_to_prompt(sample_perturbed_question)
            else:
                prompt = data_utils.sample_to_prompt_zero_shot(sample_perturbed_question) # redundant, just returns prompt

            sample_responses = data_utils.generate_response(prompt, args)
            sample_perturbed_questions.append(sample_perturbed_question)
            sample_perturbed_responses.extend(sample_responses)
            sample_perturbed_answer_results.extend([answer] * len(sample_responses))

            print(f"\nPerturbation {idx_perturb + 1}")
            print("~" * 10)
            print(f"Perturbed Question: {sample_perturbed_question}")
            print("Responses:")
            for idx, response in enumerate(sample_responses):
                print(f"{idx + 1}. {response}")

            example_questions.append(sample_perturbed_questions)
            example_responses.append(sample_perturbed_responses)
            example_answers.append(sample_perturbed_answer_results)

        perturbed_questions.append(example_questions)
        responses.append(example_responses)
        answers.append(example_answers)

    original_questions_array = np.array(original_questions, dtype=str)
    perturbed_questions_array = np.array(perturbed_questions, dtype=str)
    responses_array = np.array(responses, dtype=str)
    answers_array = np.array(answers, dtype=str)

    print("\n" + "=" * 50)
    print("Response Generation Complete".center(50))
    print("=" * 50)

    return original_questions_array, perturbed_questions_array, responses_array, answers_array

