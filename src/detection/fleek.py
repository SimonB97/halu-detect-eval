from typing import List, Dict
from src.models.llm import BaseLlm
from src.utils.utils import split_into_sentences_spacy
import json
from src.utils.utils import print_json
from tavily import TavilyClient
from ratelimit import limits, sleep_and_retry

# LLMs: 50 calls per second
LLM_CALLS = 50
LLM_RATE_LIMIT = 1
@sleep_and_retry
@limits(calls=LLM_CALLS, period=LLM_RATE_LIMIT)
def check_llm_limit():
    ''' Empty function just to check for calls to API '''

# Search API: 20 calls per minute
SEARCH_CALLS = 10
SEARCH_RATE_LIMIT = 60
@sleep_and_retry
@limits(calls=SEARCH_CALLS, period=SEARCH_RATE_LIMIT)
def check_search_limit():
    ''' Empty function just to check for calls to API '''


class FLEEK:
    def __init__(self, llm: BaseLlm, search_api_key: str):
        """
        Initializes the method with an instance of a language model for fact extraction, question generation, and verification.
        
        Args:
            llm (BaseLlm): An instance of a language model.
        """
        self.llm = llm
        self.search_api_key = search_api_key


    def extract_facts(self, sentence: str) -> List[Dict]:
        """
        Extracts facts from a sentence using the LLM. Each fact is represented as either a flat or extended triple.
        
        For flat triples, expect dictionaries of the form:
        {"type": "flat", "subject": "subject_name", "predicate": "predicate_relation", "object": "object_info"}
        
        For extended triples, expect dictionaries with additional attributes for n-ary relations:
        {"type": "extended", "subject": "subject_name", "predicate": "predicate_relation", "attributes": [{"predicate_id": "id", "predicate_attribute": "attribute_name", "object": "attribute_value"}]}
        
        Args:
            sentence (str): The sentence from which facts are to be extracted.
            
        Returns:
            List[Dict]: A list of dictionaries representing the extracted facts.
        """
        system_message = (
            "As an expert in JSON format, your task is to extract the facts from the given statement in the form of triples (flat or extended). "
            "If the statement contains complex relations, you may need to represent them as extended triples with additional attributes. Please provide the extracted facts in JSON format.\n"
            "In the JSON standard, property names must be enclosed in double quotes, and each pair of property and value must be separated by a comma. "
            "Make sure to escape double quotes in string values. For example, \"key\": \"value\". "
            "Also, each triple should be accessible by a unique key, such as 'flat1', 'flat2', 'extended1', etc., and keys should NOT be leading with an underscore '_'.\n\n"
            "For flat triples, use the following format: {'flat1': {'type': 'flat', 'subject': 'subject_name', 'predicate': 'predicate_relation', 'object': 'object_info'}}.\n"
            "For extended triples, use the following format: {'extended1': {'type': 'extended', 'subject': 'subject_name', 'predicate': 'predicate_relation', 'attributes': [{'predicate_id': 'id', 'predicate_attribute': 'attribute_name', 'object': 'attribute_value'}, "
            "{'predicate_id': 'id', 'predicate_attribute': 'attribute_name', 'object': 'attribute_value'}, ...]}}.\n\n"
            "End your JSON with the '</s>' token."
        )
        prompt_template = "Extract the facts from the given sentence: '{}'"
        prompt = prompt_template.format(sentence)
        examples = [
            {'role': 'user', 'content': prompt_template.format("Taylor Swift is 30 years old.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Taylor Swift", "predicate": "age", "object": "30 years old"}}]'},
            {'role': 'user', 'content': prompt_template.format("John has an age of 30 and resides in New York.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "John", "predicate": "age", "object": "30"}}, {"flat2": {"type": "flat", "subject": "John", "predicate": "resides in", "object": "New York"}}]'},
            {'role': 'user', 'content': prompt_template.format("John, a software engineer, works for Google in San Francisco.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "John", "predicate": "profession", "object": "software engineer"}}, {"extended1": {"type": "extended", "subject": "John", "predicate": "works for", "attributes": [{"predicate_id": "1", "predicate_attribute": "company", "object": "Google"}, {"predicate_id": "2", "predicate_attribute": "location", "object": "San Francisco"}]}}]'},
            {'role': 'user', 'content': prompt_template.format("Mary is a doctor and has a daughter named Emma who is 5 years old.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Mary", "predicate": "profession", "object": "doctor"}}, {"flat2": {"type": "flat", "subject": "Mary", "predicate": "daughter", "object": "Emma"}}, {"flat3": {"type": "flat", "subject": "Emma", "predicate": "age", "object": "5"}}]'},
            {'role': 'user', 'content': prompt_template.format("David, a software engineer at Microsoft, recently bought a house in Seattle for $1.2 million.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "David", "predicate": "profession", "object": "software engineer at Microsoft"}}, {"extended1": {"type": "extended", "subject": "David", "predicate": "bought", "attributes": [{"predicate_id": "1", "predicate_attribute": "object", "object": "house"}, {"predicate_id": "2", "predicate_attribute": "location", "object": "Seattle"}, {"predicate_id": "3", "predicate_attribute": "price", "object": "$1.2 million"}]}}]'},
            {'role': 'user', 'content': prompt_template.format("Simon lives in Berlin, has a dog named Max, and likes gaming.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Simon", "predicate": "lives in", "object": "Berlin"}}, {"flat2": {"type": "flat", "subject": "Simon", "predicate": "dog", "object": "Max"}}, {"flat3": {"type": "flat", "subject": "Simon", "predicate": "likes", "object": "gaming"}}]'},
            {'role': 'user', 'content': prompt_template.format("Alice, a lawyer, lives in London and has two children, Bob and Charlie, who are studying in Oxford.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Alice", "predicate": "profession", "object": "lawyer"}}, {"flat2": {"type": "flat", "subject": "Alice", "predicate": "lives in", "object": "London"}}, {"flat3": {"type": "flat", "subject": "Alice", "predicate": "children", "object": "Bob, Charlie"}}, {"extended1": {"type": "extended", "subject": "Bob, Charlie", "predicate": "studying in", "attributes": [{"predicate_id": "1", "predicate_attribute": "location", "object": "Oxford"}]}}]'},
            {'role': 'user', 'content': prompt_template.format("Emma, a teacher, has a son named Jack who is a doctor and lives in Paris.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Emma", "predicate": "profession", "object": "teacher"}}, {"flat2": {"type": "flat", "subject": "Emma", "predicate": "son", "object": "Jack"}}, {"flat3": {"type": "flat", "subject": "Jack", "predicate": "profession", "object": "doctor"}}, {"flat4": {"type": "flat", "subject": "Jack", "predicate": "lives in", "object": "Paris"}}]'},
            {'role': 'user', 'content': prompt_template.format("Mike, a software engineer at Amazon, lives in Seattle and has a pet dog named Rex.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Mike", "predicate": "profession", "object": "software engineer at Amazon"}}, {"flat2": {"type": "flat", "subject": "Mike", "predicate": "lives in", "object": "Seattle"}}, {"flat3": {"type": "flat", "subject": "Mike", "predicate": "pet", "object": "Rex"}}]'},
            {'role': 'user', 'content': prompt_template.format("Sarah, a nurse, has a husband named Tom who is a firefighter and they live in Chicago with their two children, Amy and Alex.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Sarah", "predicate": "profession", "object": "nurse"}}, {"flat2": {"type": "flat", "subject": "Sarah", "predicate": "husband", "object": "Tom"}}, {"flat3": {"type": "flat", "subject": "Tom", "predicate": "profession", "object": "firefighter"}}, {"flat4": {"type": "flat", "subject": "Sarah, Tom", "predicate": "live in", "object": "Chicago"}}, {"flat5": {"type": "flat", "subject": "Sarah, Tom", "predicate": "children", "object": "Amy, Alex"}}]'},
            {'role': 'user', 'content': prompt_template.format("Robert, a pilot, has a wife named Linda who is a chef. They live in San Francisco and have a daughter named Lisa who is studying medicine at Stanford.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Robert", "predicate": "profession", "object": "pilot"}}, {"flat2": {"type": "flat", "subject": "Robert", "predicate": "wife", "object": "Linda"}}, {"flat3": {"type": "flat", "subject": "Linda", "predicate": "profession", "object": "chef"}}, {"flat4": {"type": "flat", "subject": "Robert, Linda", "predicate": "live in", "object": "San Francisco"}}, {"flat5": {"type": "flat", "subject": "Robert, Linda", "predicate": "daughter", "object": "Lisa"}}, {"extended1": {"type": "extended", "subject": "Lisa", "predicate": "studying", "attributes": [{"predicate_id": "1", "predicate_attribute": "subject", "object": "medicine"}, {"predicate_id": "2", "predicate_attribute": "location", "object": "Stanford"}]}}]'},
        ]
        for example in examples:
            example['content'] = example['content'].replace('predicate_id', 'predicateID')  # Fix for JSON property name
            example['content'] = example['content'].replace('predicate_attribute', 'predicateAttribute')  # Fix for JSON property name
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        for example in examples:
            # insert right before the latest message
            messages.insert(-1, example)
        
        response = self.llm.get_response(messages, max_tokens=2048, json_mode=True)[-1]
        # print(f"DEBUG: extract_facts - sentence: \"{sentence}\"")
        # print(f"DEBUG: extract_facts - response: {response}")

        return self.parse_json(response)
    

    def parse_json(self, response: str) -> List[Dict]:
        """
        Extracts dictionary from the response and returns a list of dictionaries representing the extracted facts.

        Args:
            sentence (str): The response containing the extracted facts.

        Returns:
            List[Dict]: A list of dictionaries representing the extracted facts.
        """
        try:
            # if no list is found, add brackets to the response
            if (response[0] != '[') and (response[1] != '['):
                response = f"[{response}]"
            _, _, response = response.partition('[')
            response, _, _ = response.rpartition(']')
            response = f"[{response}]"
            # print(f"DEBUG: parsing json - response: \"{response}\"")
            dictionary = json.loads(response)
        except ValueError as e:
            print(f"Unable to parse JSON: {response}")
            raise e

        return dictionary


    def generate_questions(self, facts: List[Dict]) -> List[str]:
        """
        Generates questions for each extracted fact. Applies different strategies based on the type of the fact (flat or extended).
        
        Args:
            facts (List[Dict]): A list of fact representations extracted from the sentence.
            
        Returns:
            List[str]: A list of questions generated for each fact in the same order as the input.
        """
        questions = []
        facts = list(facts[0].values())
        
        for fact in facts:
            if fact["type"].lower() == "flat":
                # print(f"DEBUG: generate_questions - flat fact: {fact}")
                question = self.generate_flat_triple_question(fact)
            elif fact["type"].lower() == "extended":
                # print(f"DEBUG: generate_questions - extended fact: {fact}")
                question = self.generate_extended_triple_question(fact)
            questions.append(question)

        # print(f"DEBUG: generate_questions - {len(questions)} questions generated: {questions}")
        return questions
    

    def generate_flat_triple_question(self, fact: Dict) -> str:
        """
        Generates a question for a flat triple fact using Type-aware Question Generation (TQGen).
        
        Args:
            fact (Dict): A flat triple fact representation.
            
        Returns:
            str: The question generated for the fact.
        """
        system_message = (
            "As an expert in Natural Language Processing, your task is to generate a question based on the fact. "
            "The question should be designed to elicit information about the object in the fact.\n\n"
            "To achieve this task, think step by step. Follow the following plan:\n"
            "1. Identify the type of the object in the fact.\n"
            "2. Craft a question that is relevant and contextually appropriate, making use of the identified type.\n"
            "3. Return the type and the question in JSON format. The statement (context) will not be provided together with the question, just the question on it's own. "
            "E.g., if the triple is (Facebook, makes, people socially isolated), the question should NOT be 'What effect does Facebook have on people according to the statement?', "
            "but rather 'What effect does Facebook have on peoples' social life?'"
        )
        prompt_template = "Generate a question (exactly one) based on the fact:\n\nSubject: '{subject}'\nPredicate: '{predicate}'\nObject: '{object}'"
        # fact = list(fact.values())
        fact = [fact]
        try:
            prompt = prompt_template.format(subject=fact[0]['subject'], predicate=fact[0]['predicate'], object=fact[0]['object'])
        except KeyError as e:
            raise ValueError("An Error occurred while formatting the attributes of the flat triple \"{fact}\": {e}")
        examples = [
            {'role': 'user', 'content': prompt_template.format(subject="Taylor Swift", predicate="birthdate", object="1989")},
            {'role': 'assistant', 'content': '[{"type": "Year", "question": "In which year was Taylor Swift born?"}]</s>'},
            {'role': 'user', 'content': prompt_template.format(subject="John", predicate="age", object="30")},
            {'role': 'assistant', 'content': '[{"type": "Age in years", "question": "How many years old is John?"}]</s>'},
            {'role': 'user', 'content': prompt_template.format(subject="Mary", predicate="profession", object="doctor")},
            {'role': 'assistant', 'content': '[{"type": "Profession", "question": "What is Mary\'s profession?"}]</s>'}
        ]
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        for example in examples:
            messages.insert(-1, example)

        check_llm_limit()
        response = self.llm.get_response(messages, max_tokens=2048)[-1]
        response = response
        return self.parse_json(response)[0]['question']


    def generate_extended_triple_question(self, fact: Dict) -> str:
        """
        Generates a question for an extended triple fact using Context-driven Question Generation (CQGen).
        
        Args:
            fact (Dict): An extended triple fact representation.
            
        Returns:
            str: The question generated for the fact.
        """
        fact_details = fact
        
        # Prepare the system message for guiding the LLM
        system_message = (
            "As an expert in Natural Language Processing, your task is to generate a question (exactly one) that incorporates "
            "the context provided by the attributes of an extended fact triple. The question should be specific and to the point, "
            "eliciting detailed information based on the context of the attributes. Provide the question in JSON format. "
            "The statement (context) will not be provided together with the question, just the question on it's own."
        )
        
        # Format the prompt to include the subject, predicate, and attributes of the extended triple
        try:
            attributes_formatted = ", ".join([f"{attr['predicateAttribute']}: {attr['object']}" for attr in fact_details['attributes']])
        except KeyError as e:
            try:
                if e == "attributes":
                    attributes_formatted = ", ".join([f"{attr['predicateAttribute']}: {attr['object']}" for attr in fact_details['_attributes']])
                if e == "predicateAttribute":
                    attributes_formatted = ", ".join([f"{attr['predicate_attribute']}: {attr['object']}" for attr in fact_details['attributes']])
                if e == "object":
                    attributes_formatted = ", ".join([f"{attr['predicateAttribute']}: {attr['_object']}" for attr in fact_details['attributes']])
            except KeyError as e:
                logger.error(f"An Error occurred while formatting the attributes of the extended triple.\n  fact_details:\n{fact_details}\n  error: {e}")
                raise ValueError("An Error occurred while formatting the attributes of the extended triple. Please check the logs for more information.")
        prompt_template = (
            "Generate a question based on the extended fact:\n\nSubject: '{subject}'\nPredicate: '{predicate}'\nAttributes: {attributes}\n\n"
            "The question should be designed to elicit detailed information incorporating the context of the attributes and only that."
        )
        prompt = prompt_template.format(
            subject=fact_details['subject'],
            predicate=fact_details['predicate'],
            attributes=attributes_formatted
        )
        examples = [
            {'role': 'user', 'content': prompt_template.format(subject="Taylor Swift", predicate="moved to", attributes="Los Angeles")},
            {'role': 'assistant', 'content': '[{"question": "When did Taylor Swift move to Los Angeles?"}]</s>'},
        ]
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        for example in examples:
            messages.insert(-1, example)
        
        # Get the response from the language model
        response = self.llm.get_response(messages, max_tokens=2048)[-1]#
        # print(f"DEBUG: generate_extended_triple_question - response: \"{response}\"")
        question = self.parse_json(response)[0]['question']

        return question


    def retrieve_evidence_web_search(self, questions: List[str], search_depth: str, max_results: int) -> List[List[Dict[str, str]]]:
        """
        Retrieves evidence for each question using web search, returning search result snippets.
        
        Args:
            questions (List[str]): A list of questions for which to retrieve evidence.
            search_depth (str): The search depth, as defined by Tavily's API. Either "basic" or "advanced".
            max_results (int): The maximum number of search results to retrieve for each question.
            
        Returns:
            List[List[Dict[str, str]]]: A list of lists, where each inner list contains dictionaries representing the URL and content of each search result snippet.
        """
        search_results = []
        for question in questions:
            results = self.perform_web_search(question, search_depth, max_results)
            search_results.append(results)
        return search_results


    def perform_web_search(self, query: str, search_depth: str, max_results: int) -> List[Dict[str, str]]:
        """
        Performs a web search for the query and retrieves relevant snippets of information.
        
        Args:
            query (str): The search query.
            search_depth (str): The search depth, as defined by Tavily's API. Either "basic" or "advanced".
            max_results (int): The maximum number of search results to retrieve for the query.
            
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the URL and content of each search result snippet.
        """
        check_search_limit()
        client = TavilyClient(self.search_api_key)
        try:
            resp = client.search(query, search_depth, max_results=max_results)
            context = [{"url": obj["url"], "content": obj["content"]} for obj in resp["results"]]
            return context
        except Exception as e:
            print(f"An error occurred during web search: {e}")
            return [{"ERROR": "An error occurred during web search {" + str(e) + "}"}]


    def verify_facts(self, facts: List[Dict], search_results: List[List[Dict[str, str]]]) -> List[str]:
        """
        Verifies the facts based on the evidence retrieved from web search, constructing evidence triples 
        for entailment checking.
        
        Args:
            facts (List[Dict]): The facts to be verified, extracted from the text.
            search_results (List[List[Dict[str, str]]]): The search result snippets retrieved for each fact's question.
            
        Returns:
            List[str]: The verification status for each fact.
        """
        verification_results = []
        
        # Iterate through each fact and its corresponding search results
        for fact_idx, fact in enumerate(facts):
            # Assuming facts is a list of dictionaries with 'subject', 'predicate', and 'object' keys
            fact = list(fact.values())[0]
            subject = fact['subject']
            predicate = fact['predicate']

            # print(f"DEBUG: verify_facts - fact: {fact}")
            
            # Iterate through each piece of evidence for the current fact
            for evidence in search_results[fact_idx]:
                # Construct evidence triple by using the answer as the new object
                evidence_text = evidence['content']  # Assuming 'content' holds the textual evidence
                evidence_triple = {"subject": subject, "predicate": predicate, "object": evidence_text}
                
                # Prepare prompt for LLM entailment checking
                system_message = (
                    "As an expert in checking facts for groundedness, your task is to verify the evidence against the fact. "
                    "The evidence triple is constructed from the fact and the retrieved evidence. "
                    "Answer either with 'supports' if the evidence supports the fact, 'contradicts' if not, or 'neutral'. "
                    "Please provide the verification result in JSON format."
                )

                if fact['type'] == "extended":
                    fact_type = "extended"
                else:
                    fact_type = "flat"
                prompt = self.construct_verification_prompt(fact, evidence_triple, fact_type)
                # print(f"DEBUG: verify_facts - prompt: {prompt}")
                
                # Use LLM to verify the evidence against the fact
                response = self.llm.get_response(prompt, system_message, max_tokens=512, json_mode=True)[-1]
                verification_result = self.parse_llm_response_for_verification(response)
                
                # Decision logic based on LLM response
                if verification_result == "supports":
                    verification_results.append("Likely Supported")
                    break  # If one piece of evidence supports the fact, consider it verified
            else:
                # If no evidence supports the fact, mark it as Questionable or Not Supported based on your criteria
                verification_results.append("Questionable")

        return verification_results

    def construct_verification_prompt(self, fact: Dict, evidence_triple: Dict, fact_type: str) -> str:
        """
        Constructs a prompt for the LLM to verify a fact against an evidence triple.
        
        Args:
            fact (Dict): The original fact.
            evidence_triple (Dict): The evidence triple constructed from web search results.
            fact_type (str): The type of the fact, either "flat" or "extended".
            
        Returns:
            str: A prompt for the LLM.
        """
        if fact_type == "flat":
            return f"Given the fact (Subject: {fact['subject']}, Predicate: {fact['predicate']}, Object: {fact['object']}) " \
                f"and the evidence (Subject: {evidence_triple['subject']}, Predicate: {evidence_triple['predicate']}, " \
                f"Object: {evidence_triple['object']}), does the evidence support the fact?"
        else:
            attributes = ", ".join([f"{attr['predicateAttribute']}: {attr['object']}" for attr in fact['attributes']])
            return f"Given the fact (Subject: {fact['subject']}, Predicate: {fact['predicate']}, Attributes: {attributes}) " \
                f"and the evidence (Subject: {evidence_triple['subject']}, Predicate: {evidence_triple['predicate']}, " \
                f"Object: {evidence_triple['object']}), does the evidence support the fact?"

    def parse_llm_response_for_verification(self, response: str) -> str:
        """
        Parses the LLM response to extract the verification result.
        
        Args:
            response (str): The LLM's response to a verification prompt.
            
        Returns:
            str: 'supports' if the evidence supports the fact, 'contradicts' if not, or 'neutral'.
        """
        if "supports" in response:
            return "supports"
        elif "contradicts" in response:
            return "contradicts"
        else:
            return "neutral"



    def get_hallucination_score(self, response: tuple) -> List[str]:
        """
        Splits a given response into sentences and processes each sentence through the complete FLEEK methodology: fact extraction, question generation, evidence retrieval, and verification.
        
        Args:
            response (tuple): The response tuple containing tokens, logprobs, linear probabilities, and the complete response string.

        Returns:
            Dict: A dictionary with sentences as keys and their hallucination scores as values.
        """
        # with splitting into sentences
        # sentences = split_into_sentences_spacy(response[-1])
        # response_scores = []

        # for sentence in sentences:
        #     facts = self.extract_facts(sentence)
        #     print(f"DEBUG: get_hallucination_score - facts:")
        #     print_json(facts)
        #     questions = self.generate_questions(facts)
        #     print(f"DEBUG: get_hallucination_score - questions:")
        #     print_json(questions)
        #     search_results = self.retrieve_evidence_web_search(questions, "advanced", max_results=10)
        #     print(f"DEBUG: get_hallucination_score - search_results:")
        #     print_json(search_results)
        #     verification_results = self.verify_facts(facts, search_results)
        #     print(f"DEBUG: get_hallucination_score - verification_results:")
        #     print_json(verification_results)
        #     response_scores.append(verification_results)

        # without splitting into sentences
        facts = self.extract_facts(response[-1])
        # print(f"DEBUG: get_hallucination_score - facts:")
        # print_json(facts)
        questions = self.generate_questions(facts)
        # print(f"DEBUG: get_hallucination_score - questions:")
        # print_json(questions)
        search_results = self.retrieve_evidence_web_search(questions, "advanced", max_results=10)
        # print(f"DEBUG: get_hallucination_score - search_results:")
        # print_json(search_results)
        verification_results = self.verify_facts(facts, search_results)
        # return as dictionary
        # turn into dictionary with one score
        response_scores = {}
        response_scores[response[-1]] = {"score": verification_results[0]}

        return response_scores
