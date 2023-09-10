import torch
from torch import nn
import numpy as np
from modules.prompt.caco_cot import *
import requests
import re
import random
from modules.prompt.scienceqa_base_prompt import *


class CoT(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def get_single_run_claude(self, cot_prompt, prompt): 
        """
        anthropic = Anthropic(api_key=args.key)
        response = anthropic.completions.create(
            model="claude-1",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens_to_sample=3000,
            prompt=f"{HUMAN_PROMPT} {prompt+cot_prompt} {AI_PROMPT}",
        )

        return response.completion
        """
        pass

    def get_single_run_gpt(self, cot_prompt, prompt): 
        headers = {
                'Authorization': f'Bearer ' + self.cfgs["api_key"], 
                'Content-Type': 'application/json'
                }

        response = requests.post(self.cfgs["api_url"], 
                                headers=headers, 
                                json={
                                    "model": self.cfgs["llm"],
                                    "messages": [
                                        {'role': 'user', 'content': prompt}, 
                                        ],
                                    "temperature": self.cfgs["temperature"],
                                    "top_p": self.cfgs["top_p"],
                                    }).json()
        
        return response['choices'][0]['message']['content']

    def filter_output(self, output, choice):

        tags = re.findall(r'<Answer>(.*?)</Answer>', output)
        if len(tags) == 0:
            answer = output
        else:
            answer = tags[-1]
        
        num_choice = len(choice)
        # print(choice)
        if len(answer) == 1 and answer in ['A', 'B', 'C', 'D', 'E'][:num_choice]:
            return answer[0]
        
        elif answer in choice:
            return ['A', 'B', 'C', 'D', 'E'][choice.index(answer)]
        
        elif 'cannot' in answer.lower():
            return ['A', 'B', 'C', 'D', 'E'][random.choice(range(num_choice))]
        
        else: 
            answer = [a for a in ['A', 'B', 'C', 'D', 'E'][:num_choice] if a in answer]
            # print(answer)
            if len(answer) == 1: 
                return answer[-1]
            elif len(answer) > 1:
                return answer[random.choice(range(len(answer)))]
            else:
                return ['A', 'B', 'C', 'D', 'E'][random.choice(range(num_choice))]

    def build_prompt_sciqa(self, problems, qid):
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], False)
        choice = get_choice_text(problems[qid], self.cfgs["options"])
        answer = get_answer(problems[qid], self.cfgs["options"])
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])
        test_example = create_one_example(self.cfgs["prompt_format"],
                                        question,
                                        context,
                                        choice,
                                        answer,
                                        lecture,
                                        solution,
                                        test_example=True)
        # print(test_example)
        # exit(0)
        method = self.cfgs["method"]
        if method == 'base':
            prompt = base_prompt + test_example
        elif method in ['zeroshot-cot', 'zs-self-consistent', 'zs-complexity']:
            prompt = zeroshot_cot_prompt + test_example
        elif method in ['oneshot-cot', 'os-self-consistent', 'os-complexity']:
            prompt = one_shot_cot_prompt + test_example # cot_prompt + 
        elif method == 'caco_cot':
            prompt = f"\"\"\"\n{test_example}\n\"\"\""
        elif method == 'least-to-most':
            prompt = least_to_most_decomposer_prompt + \
            f"""
            Now, provide the sub-questions for the question: 
            Q: 
                Judge whether the sentense is true or false: {test_example}"""

        else:
            raise NotImplementedError('Method not implemented')
        return prompt

    def __call__(self, qid, prompt, choice):
        get_single_run = self.get_single_run_gpt
        prompt = ''
        cot_prompt = ''
        if 'self-consistent' in self.cfgs["method"] or 'complexity' in self.cfgs["method"]:
            answers = []
            outputs = ''
            n_chains = []
            for i in range(10): 
                cot_prompt = 'Let\'s think step by step. '
                response = get_single_run(cot_prompt, prompt)
                output = response

                n_chain = output.count('\n') - output.count('\n\n') + 1
                n_chains.append(n_chain) # \n or \n\n + 1 as number of chains
                outputs += '\n' + '-'*50 + f'\n\nRound {i+1} #Chain {n_chain} \n Output: {output}'
                answer = self.filter_output(output, choice)
                answers.append(answer)

                # print(answer)

                
            if 'complexity' in self.cfgs["method"]: # filter out those with a shorter reasoning chain
                longchain_indices = np.argsort(n_chains)[-int(0.6*len(n_chains)):] # indices of those with more chains
                answers = np.array(answers)[longchain_indices]

            answer = max(set(answers), key=list(answers).count)
            output = outputs
            # print(answer, output)

        elif self.cfgs["method"] in ['zeroshot-cot']:
            output = get_single_run('Let\'s think step by step. ', prompt)
            answer = self.filter_output(output, choice)

        elif self.cfgs["method"] in ['base', 'oneshot-cot', 'zeroshot-cot']:
            output = get_single_run('', prompt)
            answer = self.filter_output(output, choice)

        elif self.cfgs["method"] == 'least-to-most':
            # PROBLEM DECOMPOSITION
            response = get_single_run('', prompt)
            output = response
            
            subquestion_list = re.findall(r'<subquestion>(.*?)</subquestion>', output)

            original_question = prompt.split('Judge whether the sentense is true or false: ')[-1]
            messages = f"{least_to_most_subq_solver_prompt}Now, the context is: \n{original_question}"
            # PROBLEM SOLVING {original_question } {AI_PROMPT}
            for subq in subquestion_list: 
                messages += '\nQ: ' + subq
                
                ans = get_single_run('', messages)
                if 'A:' not in ans:
                    messages += '\nA: ' + ans + '\n\n'
                else:
                    messages += '\nA:' + ans.split('A:')[1] + '\n\n'
            
            messages += f"\nQ: Is the sentense true or false, \"{original_question}\"? Your reply MUST include '<Answer>True</Answer>' or '<Answer>False</Answer>' at the end."
            ans = get_single_run('', messages)
            messages += '\nA: ' + ans + '\n\n'
            answer = self.filter_output(ans, choice)
            output = messages.split(f'Now, the context is: \n{original_question}')[1]
            
        return qid, answer, output
