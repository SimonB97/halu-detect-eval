
# Evaluation of Hallucination Detection Methods for LLMs


## Notes regarding method implementations

- **LBHD** (Logit-based Hallucination Detection):
    - using "Instructing the Model" method to identify key concepts (for now), as no extra tool needed
    - not clear how the authors aggregated concept scores into sentence scores -> using same strategy for aggregation to sentence scores as for concept scores
    - using same llm for concept extraction as for response generation, to simulate real-world scenario

- **LM sv LM**:
    - prompts and inspiration from [github.com/maybenotime/PHD/LMvsLM_replicate](https://github.com/maybenotime/PHD/tree/main/LMvsLM_replicate)
    - 0 = true statement, 1 = hallucination, 0.5 = unsure
    - currently hardcoded max turns = 5 (as in original paper)

- **FLEEK**:
    - only using web search (Tavily) for evidence retrieval
    - custom prompts (because original ones not given in paper)
    - 6-shot for fact extraction step, to better follow JSON format (instead of 5-shot, as in original paper; authors noted difficulties with 5-shot)
    - mistralai/Mistral-7B-Instruct-v0.2 not able to follow JSON format for extracting facts as triples
    - hardcoded max results for web search to 10 (5 in original paper), since Knowledge Graph (KG) search is not used
    - unified evidence classification (supported/unsupported) and fact verification (true/false) into one step
    - only returns binary classification (supported/unsupported) for sentences (original paper used higher granularity)


## Notes on datasets

- **NQ_Open**:
    - 8-shot answer generation

- **WikiAQ**:
    - 0-shot answer generation

- hardcoded ratelimit for api calls at 50 per second (for FLEEK, additional rate limit for web search = 10 per minute)


## General Implementation Notes

**JSON Mode**:
    - used for FLEEK, not necessary but can improve robustness. Working with OpenAI models and some TogetherAI models (see [together.ai JSON docs](https://docs.together.ai/docs/json-mode))

**Results**:
    - saved in `results` folder
    - saved after:
        - answer generation
        - hallucination detection
        - 