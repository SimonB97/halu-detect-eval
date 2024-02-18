"""Logit-based Hallucination Detection (LBHD) method."""

from models.llm import BaseLlm
import numpy as np

class LBHD:
    def __init__(self, llm: BaseLlm):
        self.llm = llm

    def get_logit(self, message: str, system_message: str = None, temperature: float = 0.0) -> np.ndarray:
        """Get logit-based probabilities for each token in the response."""
        _, logprobs, _ = self.llm.get_response(message, system_message, temperature=temperature, return_logprobs=True)
        return np.array(logprobs)

    def get_hallucination_score(self, message: str, system_message: str = None, temperature: float = 0.0, variants: list[str] = None) -> float:
        """Get response and hallucination score."""
        linear_probabilities = self.get_logit(message, system_message, temperature)
        possible_variants = ["min", "max", "avg"]
        scores = []
        for variant in variants:
            if variant not in possible_variants:
                raise ValueError(f"Variant {variant} not in {possible_variants}")
            if variant == "min":
                score = np.min(linear_probabilities)
            elif variant == "max":
                score = np.max(linear_probabilities)
            else:
                score = np.mean(linear_probabilities)
            scores.append({variant: score})
        return scores


def identify_concepts(llm: BaseLlm, statement: str) -> list[str]:
    """Identify key concepts in statement."""
    prompt = f"{statement}\n\nIdentify all the important keyphrases from the above sentence and return a comma separated list."
    response = llm.get_response(prompt.format(statement=statement))[-1]
    print(f"DEBUG: response: {response}")