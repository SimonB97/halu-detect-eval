from typing import List, Dict
from src.models.llm import BaseLlm


class FLEEK:
    def __init__(self, llm: BaseLlm):
        """
        Initializes the method with an instance of a language model for fact extraction, question generation, and verification.
        
        Args:
            llm (BaseLlm): An instance of a language model.
        """
        self.llm = llm


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
        
        Implementation Detail:
            - Use the LLM to parse the sentence and identify entities, relations, and attributes.
            - Depending on the complexity of the information in the sentence, construct flat or extended triple representations.
            - This could involve prompting the LLM with specific instructions to identify and structure these facts.
        """
        return self.llm.extract_facts(sentence)


    def generate_questions(self, facts: List[Dict]) -> List[str]:
        """
        Generates questions for each extracted fact. Applies different strategies based on the type of the fact (flat or extended).
        
        Args:
            facts (List[Dict]): A list of fact representations extracted from the sentence.
            
        Returns:
            List[str]: A list of questions generated for each fact.
        
        Implementation Detail:
            - For flat triples, use Type-aware Question Generation (TQGen) that focuses on generating questions by understanding the 'type' of object.
            - For extended triples, use Context-driven Question Generation (CQGen) that incorporates context to form more precise and relevant questions.
            - This may involve crafting prompts for the LLM that guide it to generate questions based on the structure and content of the fact.
        """
        questions = []
        for fact in facts:
            if fact["type"] == "flat":
                question = self.llm.generate_question_tqgen(fact)
            elif fact["type"] == "extended":
                question = self.llm.generate_question_cqgen(fact)
            questions.append(question)
        return questions


    def retrieve_evidence_web_search(self, questions: List[str]) -> List[List[str]]:
        """
        Retrieves evidence for each question using web search, returning search result snippets.
        
        Args:
            questions (List[str]): A list of questions for which to retrieve evidence.
            
        Returns:
            List[List[str]]: A list of lists, where each inner list contains snippets of search results for a question.
        
        Implementation Detail:
            - Implement web search queries using an API or scraping technique.
            - For each question, retrieve the top-k search results and extract relevant snippets that might contain answers.
            - Ensure the search query is well-formed to maximize the relevance of the returned results.
        """
        search_results = []
        for question in questions:
            results = self.perform_web_search(question)
            search_results.append(results)
        return search_results


    def perform_web_search(self, query: str) -> List[str]:
        """
        Performs a web search for the query and retrieves relevant snippets of information.
        
        Args:
            query (str): The search query.
            
        Returns:
            List[str]: A list of snippets from the search results.
        
        Implementation Detail:
            - This function needs to perform an actual web search. You can use APIs provided by search engines like Bing or Google.
            - Process the search results to extract and return the most relevant snippets of information that answer the query.
            - Be mindful of rate limits and API costs if using a commercial search API.
        """
        # Placeholder for web search logic
        return ["Search result snippet 1", "Search result snippet 2"]  # Example results


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


    def process_sentence(self, sentence: str) -> List[str]:
        """
        Processes a given sentence through the complete FLEEK methodology: fact extraction, question generation, evidence retrieval, and verification.
        
        Args:
            sentence (str): The sentence to be processed.
            
        Returns:
            List[str]: The verification results for each fact extracted from the sentence.
        """
        facts = self.extract_facts(sentence)
        questions = self.generate_questions(facts)
        search_results = self.retrieve_evidence_web_search(questions)
        verification_results = self.verify_facts(facts, search_results)
        return verification_results
