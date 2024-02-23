"""Logit-based Hallucination Detection (LBHD) method."""

from src.models.llm import BaseLlm
from src.utils.utils import split_into_sentences_spacy
import numpy as np
import re
import traceback

class LBHD:
    def __init__(self, llm: BaseLlm):
        self.llm = llm
        self.possible_variants = ["avg", "normalized_product", "min"]

    def get_hallucination_score(self, response: tuple, variants: list[str] = None) -> dict:
        """Calculate hallucination score(s) for each sentence in the response and each concept per sentence.

        Args:
            response (tuple): The response tuple containing tokens, logprobs, linear probabilities, and the complete response string.
            variants (list[str], optional): The list of variants to calculate the hallucination score for. Defaults to None, which uses all possible variants.

        Returns:
            dict: A dictionary with sentences as keys and their hallucination scores as values.
        """
        # # print(f"DEBUG: Response response:\n- Tokens:", response[0], "\n- Logprobs:", response[1], "\n- Linear probabilities:", response[2], "\n- Full text:", response[3])
        sentences = split_into_sentences_spacy(response[-1])
        response_scores = {}

        for sentence in sentences:
            # print(f"Processing sentence: {sentence}")
            try:
                key_concepts = self.identify_concepts(self.llm, sentence)
            except Exception as e:
                print(f"Error identifying concepts for sentence: {sentence}. Error: {str(e)}")
                continue

            try:
                concept_probabilities = self.get_token_probabilities(key_concepts, response)
            except Exception as e:
                print(f"Error getting token probabilities for key concepts: {key_concepts}. Error: {str(e)}")
                continue

            variants = variants if variants else self.possible_variants
            scores_per_concept = []

            for concept, probabilities in concept_probabilities.items():
                try:
                    scores = self.get_substring_score(variants, probabilities)
                    scores_per_concept.append({concept: scores})
                except Exception as e:
                    print(f"Error getting substring score for concept: {concept}. Error: {str(e)}")
                    continue

            try:
                sentence_probabilities = self.get_token_probabilities([sentence], response)
                # print(f"DEBUG: Sentence probabilities: {list(sentence_probabilities.items())[0]}")
                sentence_scores = self.get_substring_score(variants, list(sentence_probabilities.values())[0])
            except Exception as e:
                print(f"Error getting sentence scores for sentence: {sentence}. Error: {str(e)}")
                traceback.print_exc()
                continue

            response_scores[sentence] = {"score": dict(sentence_scores, **{"concepts": scores_per_concept})}

        return response_scores
    
    def get_substring_score(self, variants, probabilities):
        scores = {}
        for variant in variants:
            if variant not in self.possible_variants:
                raise ValueError(f"Variant {variant} not in {self.possible_variants}")
            if variant == "avg":
                score = np.mean(list(probabilities.values()))
            elif variant == "normalized_product":
                score = self.normalized_product(probabilities)
            elif variant == "min":
                score = np.min(list(probabilities.values()))
            scores[variant] = score
        return scores

    def normalized_product(self, probabilities):
        adjusted_probs = [max(p, 1e-30) for p in probabilities.values()]
        log_probs = np.log(adjusted_probs)
        score = np.exp(np.sum(log_probs) / len(adjusted_probs))
        return score


    @staticmethod
    def identify_concepts(llm: BaseLlm, statement: str) -> list[str]:
        """Identify key concepts in statement.

        Args:
            llm (BaseLlm): The language model used for concept identification.
            statement (str): The statement to identify key concepts from.

        Returns:
            list[str]: The list of identified key concepts.

        """
        prompt = f"{statement}\n\nIdentify all the important keyphrases from the above sentence and return a comma separated list, using only contingous exact wordings existing in the text.  Use this exact format (including square brackets):\n\n[keyphrase1, keyphrase2, keyphrase3, ...]"
        system_message = "You are an expert in following guidelines for structuring text and identifying key concepts. Identify the key concepts in the given text."
        prompt = prompt.format(statement=statement)
        response = llm.get_response(prompt, system_message)[-1]

        # print(f"DEBUG: Concept identification response: {response}")

        # Remove brackets and any leading or trailing whitespace
        try:
            response = response.split("[")[1].split("]")[0].strip()
        except IndexError:
            print("Warning: No list found in response. Concept identification may be inaccurate.")
            response = response
        try:
            response = [item.strip() for item in response.split(",")]
        except Exception as e:
            print(f"Warning: Error occurred while splitting the response. Error: {str(e)}")
            response = []

        return response


    def get_token_probabilities(self, token_strings: list[str], response: tuple) -> dict:
        tokens, _, probabilities, _ = response
        string_probabilities = {}

        # Preprocess the concepts to match the tokenization format
        token_strings = [string.replace(' ', '').replace('.', '').replace(',', '').replace('"', '') for string in token_strings]

        for string in token_strings:
            string_tokens = []
            string_probs = []
            closest_matches = []

            # Find all possible sequences of tokens that match the string
            for i in range(len(tokens)):
                for j in range(i, len(tokens)):
                    sequence = ''.join(token.strip().replace('.', '').replace(',', '').replace('"', '') for token in tokens[i:j+1])
                    if sequence == string:
                        string_tokens = tokens[i:j+1]
                        string_probs = probabilities[i:j+1]
                    elif len(closest_matches) < 2:
                        closest_matches.append(sequence)
                    elif len(closest_matches) == 2:
                        closest_matches.sort(key=lambda x: abs(len(x) - len(string)))
                        if abs(len(sequence) - len(string)) < abs(len(closest_matches[0]) - len(string)):
                            closest_matches[0] = sequence

            if string_tokens:
                string = ''.join(string_tokens).strip().replace('.', '').replace(',', '').replace('"', '')
                string_probabilities[string] = dict(zip(string_tokens, string_probs))
            else:
                print(f"Warning: Concept '{string}' not found in response tokens.")
                print(f"Closest matches: {closest_matches}")

        return string_probabilities