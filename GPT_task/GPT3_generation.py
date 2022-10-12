from dataclasses import dataclass
from sys import prefix
from typing import List
import os
from helper import *
import openai

with open('/home/zeyi/key.txt') as f:
    key = f.read()
openai.api_key = key


class GPTppl():
    def __init__(self,device):

        self.model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.device = torch.device(device)
        self.model.to(self.device)

    def calculate_ppl(self,composed_rules):
        composed_rules = list(set(composed_rules))
        features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(composed_rule))) for composed_rule in composed_rules]
        ppl_all = []

        self.model.eval()

        with torch.no_grad():
            for index,feature in enumerate(features):
                feature = feature.to(self.device)
                loss = self.model(
                    feature,
                    labels = feature,
                    return_dict = True
                ).loss
                ppl_all.append(math.exp(loss.item()))

        return ppl_all




def filter_by_format(input,outputs,constraints,mask = '[mask]', no_filter = False):

    assert mask in input,'input format is wrong'

    selected_pattern_outputs = []
    selected_pattern_outputs_part = []
    index_mask = input.index(mask)

    prefix_end = index_mask
    suffix_start = len(input) - (index_mask + len(mask))


    for output in outputs:
    #   filter those not follow the mask pattern
        if no_filter:
            constraint_generation = output[prefix_end:-suffix_start]
            selected_pattern_outputs.append(output)
            selected_pattern_outputs_part.append(constraint_generation)

        else:

            if output[:prefix_end] == input[:prefix_end] and output[-suffix_start:] == input[-suffix_start:]:
                # output_words = output.replace(',','').split(' ')
                constraint_generation = output[prefix_end:-suffix_start]


                clause_states = []

                for constraint in constraints:

                    #  filter those not follow the constraints
                    clause_satisified = False

                    for concept in constraint:
                        if concept in constraint_generation or (concept[0].upper() + concept[1:]) in constraint_generation:
                            clause_satisified = True
                            break


                    clause_states.append(clause_satisified)

                if all(clause_states):
                    selected_pattern_outputs.append(output)
                    selected_pattern_outputs_part.append(constraint_generation)


    return selected_pattern_outputs,selected_pattern_outputs_part





@dataclass
class PromptConfig:
    engine: str = "text-davinci-002"
    max_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 1
    logprobs: int = 0
    n: int = 5
    echo: bool = False




class PromptWrapper:
    def __init__(self, prefix: str, no_filter = False):
        self.prefix = prefix
        self.negation_config = PromptConfig()
        self.ppl = GPTppl('cuda')
        self.no_filter = no_filter


    def prompt_generation(self,input, inflection_constraint, lemma_constraint,needed_count):


        prompt_str = self.create_prompt(input,lemma_constraint)


        response = openai.Completion.create(
            prompt=prompt_str,
            **self.negation_config.__dict__,
        )
        target = self.filter_generations(response.choices,input,inflection_constraint,needed_count)
        return target


    def filter_generations(self,explanations,input,constraints,needed_count):
        # Extract string explanations

        _explanations = []
        for explanation in explanations:
            text = explanation.text
            if text[-1] != '.':
                text += '.'
            _explanations.append(text.strip())


        _explanations = list(set(_explanations))


        filtered_explanations,filtered_explanations_part = filter_by_format(input,_explanations,constraints,no_filter = self.no_filter)


        if len(filtered_explanations) >= needed_count:
            needed_indexs = sorted(range(len(filtered_explanations)), key= lambda i :self.ppl.calculate_ppl(filtered_explanations)[i])[:needed_count]
            needed_explanations = [exp for (i,exp) in enumerate(filtered_explanations) if i in needed_indexs]
            needed_explanations_part = [exp for (i,exp) in enumerate(filtered_explanations_part) if i in needed_indexs]
        else:

            needed_explanations = []
            needed_explanations_part = []

        return needed_explanations,needed_explanations_part


    def create_prompt(self, input: str, constraints: str):
        return f"{self.prefix}\n" \
               f"Input: {input}\n" \
               f"Constraints: {constraints}\n"\
               f"Output:"
