from dataclasses import dataclass
from sys import prefix
from typing import List
import os
from helper import *



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




def filter_by_format(input,outputs,constraint,mask = '[mask]'):

    assert mask in input,'input format is wrong'

    selected_pattern_outputs = []
    index_mask = input.index(mask)
    if index_mask != 0:
        prefix_end = index_mask-1
    else:
        prefix_end = index_mask
    suffix_start = len(input) - (index_mask + len(mask))

    for output in outputs:
    #   filter those not follow the mask pattern
        if output[:prefix_end] == input[:prefix_end] and output[-suffix_start:] == input[-suffix_start:]:
            # output_words = output.replace(',','').split(' ')
            constraint_generation = output[prefix_end:-suffix_start]


            clause_states = []
            for concept in constraint.replace(' ','').split(','):

    #  filter those not follow the constraints
                clause_satisified = False
                if concept in constraint_generation or (concept[0].upper() + concept[1:]) in constraint_generation:
                    clause_satisified = True


                clause_states.append(clause_satisified)



            if all(clause_states):
                selected_pattern_outputs.append(output)


    return selected_pattern_outputs





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


    def prompt_generation(self, input: str, constraints: str):


        prompt_str = self.create_prompt(input,constraints)

        
        response = openai.Completion.create(
            prompt=prompt_str,
            **self.negation_config.__dict__,
        )
        target = self.filter_generations(response.choices,input,constraints)
        return target


    def filter_generations(self,explanations: List,input,constraints):
        # Extract string explanations
        try:
            filtered_explanations = [
                explanation.text.strip() for explanation in explanations
            ]
            filtered_explanations = list(set(filtered_explanations))
            
            if not self.no_filter:
                filtered_explanations = filter_by_format(input,filtered_explanations,constraints)

            # Filter out empty string / those not ending with "."
            filtered_explanations = list(
                filter(lambda exp: len(exp) > 0 and exp.endswith("."),
                    filtered_explanations))

            # Upper case the first letter
            filtered_explanations = [
                explanation[0].upper() + explanation[1:]
                for explanation in filtered_explanations
            ]

            if len(filtered_explanations) > 1:
                filtered_explanations = filtered_explanations[sorted(range(len(filtered_explanations)), key= lambda i :self.ppl.calculate_ppl(filtered_explanations)[i])[0]]

            elif len(filtered_explanations) == 1:
                filtered_explanations = filtered_explanations[0]

            elif len(filtered_explanations) == 0:
                filtered_explanations = ''
            return filtered_explanations

        except:
            return ''

    def create_prompt(self, input: str, constraints: str):
        return f"{self.prefix}\n" \
               f"Input: {input}\n" \
               f"Constraints: {constraints}\n"\
               f"Output:"
