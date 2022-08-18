from dataclasses import dataclass
from typing import List
from helper import *

with open('./key.txt') as f:
    key = f.read()
openai.api_key = key


@dataclass
class PromptConfig:
    engine: str = "text-davinci-002"
    max_tokens: int = 64
    temperature: float = 1
    top_p: float = 1
    logprobs: int = 5
    n: int = 3
    echo: bool = False


class PromptWrapper:
    def __init__(self, negation_prefix: str):
        self.negation_prefix = negation_prefix
        self.negation_config = PromptConfig(
            temperature=0,
            top_p=1,
            logprobs=0,
            n =2
        )

    def prompt_negation(self, target: str):
        """
        Generate E_tilde given E
        """
        prompt_str = self.create_negation_prompt(target)
        response = openai.Completion.create(
            prompt=prompt_str,
            **self.negation_config.__dict__,
        )
        neg_target = PromptWrapper.filter_generations(response.choices)[0]
        return neg_target

    @staticmethod
    def filter_generations(explanations: List):
        # Extract string explanations
        filtered_explanations = [explanation.text.strip() for explanation in explanations]

        # Filter out empty string / those not ending with "."
        filtered_explanations = list(filter(lambda exp: len(exp) > 0 and exp.endswith("."), filtered_explanations))

        # Upper case the first letter
        filtered_explanations = [explanation[0].upper() + explanation[1:] for explanation in filtered_explanations]

        # If there's none left, just add the first one
        if len(filtered_explanations) == 0:
            filtered_explanations.append(explanations[0].text.strip())

        # Remove duplicates
        filtered_explanations = list(dict.fromkeys(filtered_explanations))

        return filtered_explanations

    def create_negation_prompt(self, proposition: str):
        return f"{self.negation_prefix}\n" \
               f"A: {proposition}\n" \
               f"B: The statement is false."

demonstration_1 = "Given a statement, negate it to create a new sentence.\n"\
"A: To see stars at night, it is better to turn on the lights.\n"\
"B: The statement is false. To see stars at night, it is better not to turn on the lights.\n"\
"A: Falling objects cannot accelerate beyond a certain speed.\n"\
"B: The statement is false. Falling objects can accelerate beyond a certain speed.\n"\
"A: People put a number before their names.\n"\
"B: The statement is false. People do not put a number before their names."


wrapper = PromptWrapper(demonstration_1)
tmp = wrapper.prompt_negation('PersonX is seen as confident')
print(tmp)
