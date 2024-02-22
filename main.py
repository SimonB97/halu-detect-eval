from src.detection.lbhd import LBHD
from src.detection.lm_v_lm import LMvLM
from src.detection.fleek import FLEEK
from src.models.llm import OpenAILlm, TogetherAILlm, BaseLlm
from src.data.load_data import load_datasets, prepare_data
from datasets import load_dataset
import pandas as pd
import os
from multiprocessing import Pool
from ratelimit import RateLimitException
import time
from ratelimit import limits, sleep_and_retry


# 5000 calls per minute
CALLS = 50
RATE_LIMIT = 1

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    ''' Empty function just to check for calls to API '''

class Evaluation:
    def __init__(self, llm: BaseLlm):
        self.llm = llm
    

    def get_llm_answers(self, pool, data: pd.DataFrame, system_message: str = None, examples: list[dict] = None, parallel: bool = False, logprobs: bool = False, temperature: float = 0.0):
        if parallel:
            print(f"Getting LLM answers in parallel for {len(data)} requests...")
            with Pool() as pool:
                llm_answers = pool.starmap(self.get_llm_response, [(row, system_message, examples, logprobs, temperature) for _, row in data.iterrows()])
        else:
            print(f"Getting LLM answers sequentially for {len(data)} requests...")
            llm_answers = []
            for index, row in data.iterrows():
                llm_answers.append(self.get_llm_response(row, system_message, examples, logprobs, temperature))
        return llm_answers


    def get_llm_response(self, row, system_message=None, examples=None, logprobs=False, temperature=0.0):
        while True:
            try:
                check_limit()  # check rate limit before making a call
                if examples:
                    messages = examples.copy()
                    messages.append({"role": "user", "content": row["prompt"]})
                    return self.llm.get_response(messages, system_message if system_message else None, return_logprobs=logprobs, temperature=temperature)
                else:
                    return self.llm.get_response(row["prompt"], system_message if system_message else None, return_logprobs=logprobs, temperature=temperature)
            except RateLimitException:
                time.sleep(1)  # wait for 1 second before retrying


    def create_answers_df(self, data: pd.DataFrame, llm_answers: list[str]):
        print("Appending LLM answers to dataframe...")
        answers = data.copy()
        column_names = answers.columns
        llm_answer_cols = [col for col in column_names if "llm_answer" in col]

        if llm_answer_cols:
            max_answer_idx = max([int(col.split('__')[-1]) for col in llm_answer_cols if "__" in col], default=0)
            new_col = f"llm_answer__{max_answer_idx + 1}"
        else:
            new_col = "llm_answer"

        answers[new_col] = llm_answers
        return answers
        

    def get_NQopen_messages(self):
        system_message = (
            "As an expert in Succinct Answering of Questions, your task is to answer the question given the context. "
            "Provide your brief answer in form of a single sentence enclosed in square brackets. "
        )
        examples = [
            {"role": "user", "content": "where did immigrants enter the us on the west coast"},
            {"role": "assistant", "content": "['Immigrants entered the US on the west coast at Angel Island Immigration Station, San Francisco Bay']</s>"},
            {"role": "user", "content": "who won the 2017 ncaa mens basketball tournament"},
            {"role": "assistant", "content": "['North Carolina won the 2017 NCAA Men’s Basketball Tournament']</s>"},
            {"role": "user", "content": "oklahoma's 10 geographic regions are defined by surface features called"},
            {"role": "assistant", "content": "['Oklahoma’s 10 geographic regions are defined by surface features called ecological regions']</s>"},
            {"role": "user", "content": "what domain has more individuals than all other domains combined do"},
            {"role": "assistant", "content": "['the com TLD domain has more individuals than all other domains combined']</s>"},
            {"role": "user", "content": "who will take the throne after the queen dies"},
            {"role": "assistant", "content": "['Charles, the Prince of Wales, will take the throne after the Queen dies']</s>"},
            {"role": "user", "content": "what is the most important part of a computer"},
            {"role": "assistant", "content": "['the CPU is the most important part of a computer']</s>"},
            {"role": "user", "content": "where is the largest desert in the world"},
            {"role": "assistant", "content": "['Antarctica is where the largest desert in the world is']</s>"},
            {"role": "user", "content": "what college does everyone in gossip girl go to"},
            {"role": "assistant", "content": "['New York University and Columbia University are the colleges everyone in gossip girl goes to']</s>"},
        ]
        return {"system_message": system_message, "examples": examples}


    def get_XSUM_messages(self):
        system_message = (
            "As an expert in Text Summarization, your task is to summarize the article given in one sentence. "
            "Please provide a succinct, one-sentence summary of the key theme or conclusion presented in the given article."
            "Very important: Keep the sentence length to a maximum of 10 to words."
        )
        return {"system_message": system_message}





if __name__ == "__main__":
    together_bearer_token = os.getenv("TOGETHER_AUTH_BEARER_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    DEBUG = False

    llms = {
        "openai": OpenAILlm(openai_api_key, "gpt-3.5-turbo", debug=DEBUG),
        # "togetherai": TogetherAILlm(together_bearer_token, "mistralai/Mixtral-8x7B-Instruct-v0.1", debug=DEBUG)
        "togetherai": TogetherAILlm(together_bearer_token, "mistralai/Mistral-7B-Instruct-v0.2", debug=DEBUG)
    }

     # Load datasets
    datasets = load_datasets()
    prepared_data = prepare_data(datasets)
    nqopen = prepared_data["nqopen"]
    xsum = prepared_data["xsum"]

    with Pool() as pool:
        for llm_name, llm in llms.items():
            evaluation = Evaluation(llm)

            # Get LLM answers
            RANGE = 3
            nqopen_llm_answers = evaluation.get_llm_answers(
                data=nqopen.head(RANGE),
                parallel=True, 
                temperature=0.0, 
                logprobs=True,
                pool=pool,
                **evaluation.get_NQopen_messages()
            )
            xsum_llm_answers = evaluation.get_llm_answers(
                data=xsum.head(RANGE),
                parallel=True, 
                temperature=0.0, 
                logprobs=True,
                pool=pool,
                **evaluation.get_XSUM_messages()
            )

            # Create answers dataframe
            nqopen_answers = evaluation.create_answers_df(nqopen.head(RANGE), nqopen_llm_answers)
            xsum_answers = evaluation.create_answers_df(xsum.head(RANGE), xsum_llm_answers)

            # Save answers
            if not os.path.exists('results'):
                os.makedirs('results')
            nqopen_answers.to_csv(f"results/{llm_name}_nqopen_with_llm_answers.csv", index=False)
            xsum_answers.to_csv(f"results/{llm_name}_xsum_with_llm_answers.csv", index=False)