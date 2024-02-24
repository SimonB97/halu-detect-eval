
# Evaluation of Hallucination Detection Methods for LLMs


## Notes regarding method implementations

- **LBHD** (Logit-based Hallucination Detection):
    - using "Instructing the Model" method to identify key concepts (for now), as no extra tool needed
    - using same llm for concept extraction as for response generation, to simulate real-world scenario
    - concept-level scores: aggregating the aggregated scores of concept tokens instead of all concept tokens in a sentence at once ( -> e.g. avg(avg(concept.tokens) for concept in sentence_concepts))

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

- **SelfCheckGPT**:
    - 5 samples at temprature 1.0 are compared with first answer at temperature 0.0

## Notes on datasets

- **NQ_Open**:
    - 8-shot answer generation

- **WikiAQ**:
    - 0-shot answer generation

- hardcoded ratelimit for api calls at 50 per second (for FLEEK, additional rate limit for web search = 10 per minute)
- first 80 samples of each dataset's validation set used


## General Implementation Notes

- Generally using temperature 0.0 for answer generation, to get more reproducible results

**JSON Mode**:
    - used for FLEEK, not necessary but can improve robustness. Working with OpenAI models and some TogetherAI models (see [together.ai JSON docs](https://docs.together.ai/docs/json-mode))

**Results**:
    - saved in `results` folder
    - saved after:
        - answer generation
        - hallucination detection
        - 

**Evaluation**:
    - gpt-4-0613 (OpenAI) used to generate ground truth labels