"""Logit-based Hallucination Detection (LBHD) method."""

from models.llm import BaseLlm
import numpy as np

class LBHD:
    def __init__(self, llm: BaseLlm):
        self.llm = llm
        self.possible_variants = ["avg", "normalized_product", "min"]

    def get_hallucination_score(self, response, variants: list[str] = None) -> float:
        """Get response and hallucination score.

        Args:
            response (tuple): The response tuple containing tokens, attentions, probabilities, and other information.
            variants (list[str], optional): The list of variants to calculate the hallucination score. Defaults to None.

        Returns:
            float: The hallucination score.

        Raises:
            ValueError: If an invalid variant is provided.

        """
        

        key_concepts = self.identify_concepts(self.llm, response[-1])
        print(f"DEBUG: key_concepts: {key_concepts}")

        concept_probabilities = self.get_concept_probabilities(key_concepts, response)

        variants = variants if variants else self.possible_variants

        for concept, probability in concept_probabilities.items():
            scores = []
            for variant in variants:
                if variant not in self.possible_variants:
                    raise ValueError(f"Variant {variant} not in {self.possible_variants}")
                if variant == "avg":
                    score = np.mean(list(probability.values()))
                elif variant == "normalized_product":
                    # Ensure that we have no zero probabilities, as log(0) is undefined
                    adjusted_probs = [max(p, 1e-30) for p in probability.values()]
                    # Take the log of the probabilities to make multiplication a summation
                    log_probs = np.log(adjusted_probs)
                    # Sum the log probabilities and normalize
                    sum_log_probs = np.sum(log_probs)
                    # Exponentiate the average log probability to get the geometric mean
                    score = np.exp(sum_log_probs / len(adjusted_probs))
                elif variant == "min":
                    score = np.min(list(probability.values()))
                scores.append({variant: score})
            
            return 

    def identify_concepts(llm: BaseLlm, statement: str) -> list[str]:
        """Identify key concepts in statement.

        Args:
            llm (BaseLlm): The language model used for concept identification.
            statement (str): The statement to identify key concepts from.

        Returns:
            list[str]: The list of identified key concepts.

        """
        prompt = f"{statement}\n\nIdentify all the important keyphrases from the above sentence and return a comma separated list. Use the format: [keyphrase1, keyphrase2, keyphrase3, ...]"
        response = llm.get_response(prompt.format(statement=statement))[-1]
        print(f"DEBUG: response: {response}")

        # Remove brackets and any leading or trailing whitespace
        try:
            response = response.split("[")[1].split("]")[0].strip()
        except IndexError:
            print("Warning: No list found in response. Concept identification may be inaccurate.")
            response = response

        return response.split(",")

    def get_concept_probabilities(self, concepts: list[str], response: tuple) -> dict:
        """Map each concept to its tokens and their corresponding probabilities.

        Args:
            concepts (list[str]): The list of concepts to map.
            response (tuple): The response tuple containing tokens, attentions, probabilities, and other information.

        Returns:
            dict: The dictionary mapping each concept to its tokens and probabilities.

        """
        tokens, _, probabilities, _ = response
        concept_probabilities = {}

        for concept in concepts:
            # concept = concept.strip()  # Remove leading and trailing whitespace
            concept_tokens = []
            concept_probs = []

            # Find the sequence of tokens that matches the concept
            for i in range(len(tokens)):
                if ' '.join(tokens[i:i+len(concept.split())]) == concept:
                    concept_tokens = tokens[i:i+len(concept.split())]
                    concept_probs = probabilities[i:i+len(concept.split())]
                    break

            if concept_tokens:
                concept_probabilities[concept] = dict(zip(concept_tokens, concept_probs))
            else:
                print(f"Warning: Concept '{concept}' not found in response tokens.")

        return concept_probabilities