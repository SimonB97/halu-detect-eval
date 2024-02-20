from typing import List, Dict
from src.models.llm import BaseLlm
import json
from src.utils.utils import print_json
from tavily import TavilyClient



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
            "As an expert in JSON format, your task is to extract the facts from the given sentence in the form of triples. "
            "Each triple should contain a subject, predicate, and object. If the sentence contains complex relations, "
            "you may need to represent them as extended triples with additional attributes. Please provide the extracted facts in JSON format.\n"
            "In the JSON standard, property names must be enclosed in double quotes, and each pair of property and value must be separated by a comma."
        )
        prompt_template = "Extract the facts from the given sentence: '{}'"
        prompt = prompt_template.format(sentence)
        examples = [
            {'role': 'user', 'content': prompt_template.format("Taylor Swift is 30 years old.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Taylor Swift", "predicate": "age", "object": "30 years old"}}]</s>'},
            {'role': 'user', 'content': prompt_template.format("John has an age of 30 and resides in New York.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "John", "predicate": "age", "object": "30"}}, {"flat2": {"type": "flat", "subject": "John", "predicate": "resides in", "object": "New York"}}]</s>'},
            {'role': 'user', 'content': prompt_template.format("John, a software engineer, works for Google in San Francisco.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "John", "predicate": "profession", "object": "software engineer"}}, {"extended1": {"type": "extended", "subject": "John", "predicate": "works for", "attributes": [{"predicate_id": "1", "predicate_attribute": "company", "object": "Google"}, {"predicate_id": "2", "predicate_attribute": "location", "object": "San Francisco"}]}}]</s>'},
            {'role': 'user', 'content': prompt_template.format("Mary is a doctor and has a daughter named Emma who is 5 years old.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Mary", "predicate": "profession", "object": "doctor"}}, {"flat2": {"type": "flat", "subject": "Mary", "predicate": "daughter", "object": "Emma"}}, {"flat3": {"type": "flat", "subject": "Emma", "predicate": "age", "object": "5"}}]</s>'},
            {'role': 'user', 'content': prompt_template.format("David, a software engineer at Microsoft, recently bought a house in Seattle for $1.2 million.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "David", "predicate": "profession", "object": "software engineer at Microsoft"}}, {"extended1": {"type": "extended", "subject": "David", "predicate": "bought", "attributes": [{"predicate_id": "1", "predicate_attribute": "object", "object": "house"}, {"predicate_id": "2", "predicate_attribute": "location", "object": "Seattle"}, {"predicate_id": "3", "predicate_attribute": "price", "object": "$1.2 million"}]}}]</s>'},
            {'role': 'user', 'content': prompt_template.format("Simon lives in Berlin, has a dog named Max, and likes gaming.")},
            {'role': 'assistant', 'content': '[{"flat1": {"type": "flat", "subject": "Simon", "predicate": "lives in", "object": "Berlin"}}, {"flat2": {"type": "flat", "subject": "Simon", "predicate": "dog", "object": "Max"}}, {"flat3": {"type": "flat", "subject": "Simon", "predicate": "likes", "object": "gaming"}}]</s>'},
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
        
        response = self.llm.get_response(messages, max_tokens=2048)[-1]

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
            _, _, response = response.partition('[')
            response, _, _ = response.rpartition(']')
            response = f"[{response}]"
            dictionary = json.loads(response)
        except ValueError as e:
            print(f"Unable to parse JSON: {response}")
            dictionary = [{"ERROR": "Unable to parse JSON"}]

        return dictionary


    def generate_questions(self, facts: List[Dict]) -> List[str]:
        """
        Generates questions for each extracted fact. Applies different strategies based on the type of the fact (flat or extended).
        
        Args:
            facts (List[Dict]): A list of fact representations extracted from the sentence.
            
        Returns:
            List[str]: A list of questions generated for each fact in the same order as the input.
        
        Implementation Detail:
            - For flat triples, use Type-aware Question Generation (TQGen) that focuses on generating questions by understanding the 'type' of object.
            - For extended triples, use Context-driven Question Generation (CQGen) that incorporates context to form more precise and relevant questions.
            - This may involve crafting prompts for the LLM that guide it to generate questions based on the structure and content of the fact.
        """
        questions = []
        for fact in facts:
            if list(fact.values())[0]["type"].lower() == "flat":
                question = self.generate_flat_triple_question(fact)
            elif list(fact.values())[0]["type"].lower() == "extended":
                question = self.generate_extended_triple_question(fact)
            questions.append(question)
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
            "3. Return the type and the question in JSON format."
        )
        prompt_template = "Generate a question (exactly one) based on the fact:\n\nSubject: '{subject}'\nPredicate: '{predicate}'\nObject: '{object}'"
        fact = list(fact.values())
        prompt = prompt_template.format(subject=fact[0]['subject'], predicate=fact[0]['predicate'], object=fact[0]['object'])
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
        # Extract the first (and typically only) value from the fact dictionary, which contains the extended triple
        fact_details = list(fact.values())[0]
        
        # Prepare the system message for guiding the LLM
        system_message = (
            "As an expert in Natural Language Processing, your task is to generate a question (exactly one) that incorporates "
            "the context provided by the attributes of an extended fact triple. The question should be specific and to the point, "
            "eliciting detailed information based on the context of the attributes. Provide the question in JSON format."
        )
        
        # Format the prompt to include the subject, predicate, and attributes of the extended triple
        attributes_formatted = ", ".join([f"{attr['predicateAttribute']}: {attr['object']}" for attr in fact_details['attributes']])
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
        question = self.parse_json(response)[0]['question']

        return question


    def retrieve_evidence_web_search(self, questions: List[str], search_depth: str) -> List[List[str]]:
        """
        Retrieves evidence for each question using web search, returning search result snippets.
        
        Args:
            questions (List[str]): A list of questions for which to retrieve evidence.
            search_depth (str): The search depth, as defined by Tavily's API. Either "basic" or "advanced".
            
        Returns:
            List[List[str]]: A list of lists, where each inner list contains snippets of search results for a question.
        """
        search_results = []
        for question in questions:
            results = self.perform_web_search(question, search_depth)
            search_results.append(results)
        return search_results


    def perform_web_search(self, query: str, search_depth: str) -> List[Dict[str, str]]:
        """
        Performs a web search for the query and retrieves relevant snippets of information.
        
        Args:
            query (str): The search query.
            search_depth (str): The search depth, as defined by Tavily's API. Either "basic" or "advanced".
            
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the URL and content of each search result snippet.
        """
        client = TavilyClient(self.search_api_key)
        try:
            resp = client.search(query, search_depth, max_results=5)
            context = [{"url": obj["url"], "content": obj["content"]} for obj in resp["results"]]
            return context
        except Exception as e:
            print(f"An error occurred during web search: {e}")
            return [{"ERROR": "An error occurred during web search {" + str(e) + "}"}]


    def verify_facts(self, facts: List[Dict], search_results: List[List[str]]) -> List[str]:
        """
        Verifies the facts based on the evidence retrieved from web search.
        
        Args:
            facts (List[Dict]): The facts to be verified.
            search_results (List[List[str]]): The search result snippets retrieved for each fact's question.
            
        Returns:
            List[str]: The verification status for each fact, e.g., "Supported", "Unsupported".
        
        Implementation Detail:
            - For each fact, check if the corresponding search results contain information that supports or contradicts the fact.
            - This step might involve semantic analysis to compare the fact details with the information in the search results.
            - Consider using NLP techniques for entailment or contradiction detection to enhance the verification process.
        """
        verification_results = []
        for fact, results in zip(facts, search_results):
            verification_status = "Unsupported"  # Default to unsupported
            for result in results:
                if self.fact_supported_by_evidence(fact, result):
                    verification_status = "Supported"
                    break  # Assuming one piece of supporting evidence is sufficient
            verification_results.append(verification_status)
        return verification_results


    def fact_supported_by_evidence(self, fact: Dict, evidence: str) -> bool:
        """
        Determines if a piece of evidence supports a given fact.
        
        Args:
            fact (Dict): The fact being verified.
            evidence (str): A snippet of information retrieved from web search.
            
        Returns:
            bool: True if the evidence supports the fact, False otherwise.
        
        Implementation Detail:
            - Analyze the evidence to determine if it supports the fact.
            - This might involve keyword matching, checking for semantic similarity, or performing entailment analysis between the fact and evidence.
            - The level of analysis required can vary based on the complexity of the fact and the clarity of the evidence.
        """
        return "supporting information" in evidence  # Simplified example


    def get_hallucination_score(self, response: tuple) -> List[str]:
        """
        Processes a given sentence through the complete FLEEK methodology: fact extraction, question generation, evidence retrieval, and verification.
        
        Args:
            response (tuple): The response tuple containing tokens, logprobs, linear probabilities, and the complete response string. Only the string is used here.
            
        Returns:
            List[str]: The verification results for each fact extracted from the sentence.
        """
        facts = self.extract_facts(response[-1])
        questions = self.generate_questions(facts)
        search_results = self.retrieve_evidence_web_search(questions)
        verification_results = self.verify_facts(facts, search_results)
        return verification_results
