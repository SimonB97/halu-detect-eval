"""Logit-based Hallucination Detection (LBHD) method."""

from src.models.llm import BaseLlm
import numpy as np
import re

class LBHD:
    def __init__(self, llm: BaseLlm):
        self.llm = llm
        self.possible_variants = ["avg", "normalized_product", "min"]

    def get_hallucination_score(self, response, variants: list[str] = None) -> dict:
        """Calculate hallucination score(s) for each sentence in the response.

        Args:
            response (tuple): The response tuple containing tokens, logprobs, linear probabilities, and the complete response string.
            variants (list[str], optional): The list of variants to calculate the hallucination score for. Defaults to None, which uses all possible variants.

        Returns:
            dict: A dictionary with sentences as keys and their hallucination scores as values.
        """
        sentences = self.split_into_sentences(response[-1])
        response_scores = {}

        for sentence in sentences:
            print(f"Processing sentence: {sentence}")
            key_concepts = self.identify_concepts(self.llm, sentence)
            concept_probabilities = self.get_token_probabilities(key_concepts, response)

            variants = variants if variants else self.possible_variants
            scores_for_sentence = []

            for concept, probabilities in concept_probabilities.items():
                scores = []
                for variant in variants:
                    if variant not in self.possible_variants:
                        raise ValueError(f"Variant {variant} not in {self.possible_variants}")
                    if variant == "avg":
                        score = np.mean(list(probabilities.values()))
                    elif variant == "normalized_product":
                        score = self.normalized_product(probabilities)
                    elif variant == "min":
                        score = np.min(list(probabilities.values()))
                    scores.append({variant: score})
                
                scores_for_sentence.append({concept: scores})

            # calculate scores for all tokens in the sentence
            

            response_scores[sentence] = scores_for_sentence

        return response_scores

    def normalized_product(self, probabilities):
        adjusted_probs = [max(p, 1e-30) for p in probabilities.values()]
        log_probs = np.log(adjusted_probs)
        score = np.exp(np.sum(log_probs) / len(adjusted_probs))
        return score

    @staticmethod
    def split_into_sentences(text: str) -> list:
        """Split text into sentences using simple punctuation marks."""
        sentences = re.split(r'(?<=[.!?]) +', text)
        return sentences

    @staticmethod
    def identify_concepts(llm: BaseLlm, statement: str) -> list[str]:
        """Identify key concepts in statement.

        Args:
            llm (BaseLlm): The language model used for concept identification.
            statement (str): The statement to identify key concepts from.

        Returns:
            list[str]: The list of identified key concepts.

        """
        prompt = f"{statement}\n\nIdentify all the important keyphrases from the above sentence and return a comma separated list. Use the format: [keyphrase1, keyphrase2, keyphrase3, ...]"
        prompt = prompt.format(statement=statement)
        response = llm.get_response(prompt)[-1]

        # Remove brackets and any leading or trailing whitespace
        try:
            response = response.split("[")[1].split("]")[0].strip()
        except IndexError:
            print("Warning: No list found in response. Concept identification may be inaccurate.")
            response = response

        return response.split(",")


    def get_token_probabilities(self, token_strings: list[str], response: tuple) -> dict:
        """Map each string (a token-collection, in order and contigous) to its tokens and their corresponding probabilities.

        Args:
            token_strings (list[str]): The list of token-collections to map.
            response (tuple): The response tuple containing tokens, attentions, probabilities, and other information.

        Returns:
            dict: The dictionary mapping each string to its tokens and probabilities.

        """
        tokens, _, probabilities, _ = response
        string_probabilities = {}

        # Preprocess the concepts to match the tokenization format
        token_strings = [string.replace(' ', '').replace('.', '').replace(',', '') for string in token_strings]

        for string in token_strings:
            string_tokens = []
            string_probs = []

            # Find all possible sequences of tokens that match the string
            for i in range(len(tokens)):
                for j in range(i, len(tokens)):
                    sequence = ''.join(token.strip().replace('.', '').replace(',', '') for token in tokens[i:j+1])
                    # print(f"Comparing '{string}' with '{sequence}'")
                    if sequence == string:
                        string_tokens = tokens[i:j+1]
                        string_probs = probabilities[i:j+1]
                        # print(f"Matched: {string} -> {dict(zip(string_tokens, string_probs))}")

            if string_tokens:
                # concatenate the tokens to form the string back together
                # print(f"DEBUG: Tokens: {tokens}")
                # print(f"DEBUG: Concept tokens: {string_tokens}")
                string = ''.join(string_tokens).strip().replace('.', '')
                string_probabilities[string] = dict(zip(string_tokens, string_probs))
            else:
                print(f"Warning: Concept '{string}' not found in response tokens.")

        return string_probabilities