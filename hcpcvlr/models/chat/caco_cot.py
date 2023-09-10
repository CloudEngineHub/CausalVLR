import torch
from torch import nn
from modules.prompt.caco_cot import *
import requests
from .cot import CoT
from time import sleep
import numpy as np


class CaCoCoT(CoT):
    def __init__(self, cfgs):
        super().__init__(cfgs)
    

    # @retry(delay=3, tries=10, backoff=2, max_delay=120)
    def answer_review(self, question=None, choice=None, depth=1, history=''):
        get_single_run = self.get_single_run_gpt

        output_1 = get_single_run(None, our_reasoner1_prompt + question)
        answer_1 = self.filter_output(output_1, choice=choice)

        sleep(0.1)
        output_2 = get_single_run(None, our_reasoner2_prompt + question)
        answer_2 = self.filter_output(output_2, choice=choice)

        history += f" ########## Round {depth} ########## \n" + str(output_1 + '\n\n' + '-'*30 + '\n\n' + output_2 + '\n\n')

        answers = []

        if answer_1 == answer_2: 

            lst = ['A', 'B', 'C', 'D', 'E']
            lst = lst[:len(choice)]
            lst.pop(lst.index(answer_1))

            answer_cf = np.random.choice(lst)
            reviewer_prompt1 = our_reviewer_prompt + f"Now, For the question: \n\"\"\"\n{question}\n\"\"\" \n\n\nA possible solution is: \n\"\"\"\n{np.random.choice([output_1, output_2])}\n\"\"\"\n\nPlease provide your evaluation, while looking out for option ({answer_cf}). Remember, if the answer cannot be determined, make an educated guess at the end. "

            reviewer_output_3 = get_single_run(None, reviewer_prompt1)
            answer_3 = self.filter_output(reviewer_output_3, choice)

            history += f" ######## Round {depth} Review ######## \n\n" + str(reviewer_output_3 + '\n\n' + '='*30 + '\n\n')

            answers = [answer_1, answer_2, answer_3]
            max_answer = max(set(answers), key=answers.count)

            if answer_3 == answer_1: 
                history += f'\n\n{answer_1} {answer_2} {answer_3} CF: {answer_cf}  >>> all consensus \n\n'
                return answer_3, history
            elif depth >= 4: 
                history += f'\n\n{answer_1} {answer_2} {answer_3} CF: {answer_cf} >>> reach max depth\n\n'
                return np.random.choice([answer_1, answer_2,]), history
            else: 
                history += f'\n\n{answer_1} {answer_2} {answer_3} CF: {answer_cf} >>> go deeper\n\n'
                return self.answer_review(question=question, choice=choice, depth=depth+1, history=history)

        elif answer_1 != answer_2: 

            reviewer_prompt1 = our_reviewer_prompt + f"Now, For the question: \n\"\"\"\n{question}\n\"\"\" \n\n\nA possible solution is: \n\"\"\"\n{output_1}\n\"\"\"\n\nPlease provide your evaluation, while looking out for option ({answer_2}). Remember, if the answer cannot be determined, make an educated guess at the end. "
            reviewer_prompt2 = our_reviewer_prompt + f"Now, For the question: \n\"\"\"\n{question}\n\"\"\" \n\n\nA possible solution is: \n\"\"\"\n{output_2}\n\"\"\"\n\nPlease provide your evaluation, while looking out for option ({answer_1}). Remember, if the answer cannot be determined, make an educated guess at the end. "

            reviewer_output_3 = get_single_run(None, reviewer_prompt1)
            reviewer_output_4 = get_single_run(None, reviewer_prompt2)
            
            answer_3 = self.filter_output(reviewer_output_3, choice)
            answer_4 = self.filter_output(reviewer_output_4, choice)

            history += f" ######## Round {depth} Review ######## \n\n" + str(reviewer_output_3 + '\n\n' + '-'*30 + '\n\n' + reviewer_output_4 + '\n\n' + '='*30 + '\n\n')

            answers = [answer_1, answer_2, answer_3, answer_4]
            max_answer = max(set(answers), key=answers.count)

            if answer_3 == answer_4: 
                history += f'\n\n{answer_1} {answer_2} {answer_3} {answer_4} >>> evaluators\' consensus \n\n'
                return answer_3, history
            elif depth >= 4: 
                history += f'\n\n{answer_1} {answer_2} {answer_3} {answer_4} >>> reach the max depth\n\n'
                return np.random.choice([answer_1, answer_2]), history
            else: 
                history += f'\n\n{answer_1} {answer_2} {answer_3} {answer_4} >>> go deeper\n\n'
                return self.answer_review(question=question, choice=choice, depth=depth+1, history=history)


    def __call__(self, qid, prompt, choice):
        get_single_run = self.get_single_run_gpt
        
        answer, output = self.answer_review(question=prompt, choice=choice, depth=1, history='')
            
        return qid, answer, output