from src.detection import lbhd, lm_v_lm, fleek
from src.models.llm import OpenAILlm, TogetherAILlm, BaseLlm
from src.data.load_data import load_datasets, prepare_data
import pandas as pd
import os
from multiprocessing import Pool
from ratelimit import RateLimitException
import time
from ratelimit import limits, sleep_and_retry
import multiprocessing
import datetime
import numpy as np
import time
import logging


# 50 calls per second
CALLS = 50
RATE_LIMIT = 1

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    ''' Empty function just to check for calls to API '''


class Evaluation:
    def __init__(self, llm: BaseLlm):
        self.llm = llm
    

    def get_llm_answers(self, pool, data: pd.DataFrame, system_message: str = None, examples: list[dict] = None, 
                        parallel: bool = False, logprobs: bool = False, temperature: float = 0.0):
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


    def get_hallucination_scores(self, pool, data_with_answers: pd.DataFrame, detection: list[str], parallel: bool = False):
        print(f"Calculating hallucination scores for {len(data_with_answers)} answers * {len(detection)} detection methods...")
        data_with_scores = data_with_answers.copy()

        def calculate_score(method, row):
            if method == "lbhd":
                return lbhd.LBHD(self.llm).get_hallucination_score(response=row["llm_answer"])
            elif method == "lm_v_lm":
                return lm_v_lm.LMvLM(self.llm, self.llm).get_hallucination_score(response=row["llm_answer"])
            elif method == "fleek":
                return fleek.FLEEK(self.llm, tavily_api_key).get_hallucination_score(response=row["llm_answer"])

        for method in detection:
            print(f"Calculating hallucination scores for {method}...")

            column_name = f"{method}_score"
            if column_name in data_with_scores.columns:
                column_name += f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            data_with_scores[column_name] = {}
            start_time = time.time()
            if parallel:
                # TODO: fix parallel processing
                with multiprocessing.Pool(processes=pool) as p:
                    scores = p.starmap(calculate_score, [(method, row) for _, row in data_with_answers.iterrows()])
                # print(f"DEBUG: get_hallucination_scores: scores: {scores}")
                for index, score in zip(data_with_answers.index, scores):
                    data_with_scores.at[index, column_name] = score
            else:
                for index, row in data_with_answers.iterrows():
                    score = calculate_score(method, row)
                    # print(f"DEBUG: get_hallucination_scores: score: {list(score.items())[0][1]['score']}\ntype: {type(list(score.items())[0][1])}")
                    data_with_scores.at[index, column_name] = list(score.items())[0][1]["score"]
            
            logging.info(f"Time taken to calculate hallucination scores for {method}: {time.time() - start_time} seconds")

        return data_with_scores





if __name__ == "__main__":
    # Load API keys
    together_bearer_token = os.getenv("TOGETHER_AUTH_BEARER_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")  # needed for FLEEK web search

    # Set up detection methods
    detection_methods = [
                "fleek",
                "lm_v_lm", 
                "lbhd", 
            ]

    # Load LLMs
    DEBUG = False  # use to display api request details
    llms = {
            "openai": OpenAILlm(openai_api_key, "gpt-3.5-turbo", debug=DEBUG),
            # "togetherai": TogetherAILlm(together_bearer_token, "mistralai/Mixtral-8x7B-Instruct-v0.1", debug=DEBUG),
            # "togetherai_2": TogetherAILlm(together_bearer_token, "mistralai/Mistral-7B-Instruct-v0.1", debug=DEBUG),
        }
    
    # Set up logging
    logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Load datasets
    start_time = time.time()
    RANGE = 3   # minimum (RANGE * n_datasets * n_llms) requests made to web search API
    datasets = load_datasets()
    prepared_data = prepare_data(datasets)
    nqopen = prepared_data["nqopen"].iloc[:RANGE]
    xsum = prepared_data["xsum"].iloc[:RANGE]
    logging.info(f"Time taken to load datasets: {time.time() - start_time} seconds")

    # Get LLM answers  
    OVERWRITE = True
    csv_loaded_triggers = {"nqopen": False, "xsum": False}
    with Pool() as pool:
        for llm_name, llm in llms.items():
            print(f"Processing LLM: {llm.model}...")
            
            evaluation = Evaluation(llm)

            answers_paths = {
                "nqopen": f"{llm_name}_nqopen_with_answers__{llm.model}".replace("/", "_").replace("\\", "_").replace(".", "-") + ".csv",
                "xsum": f"{llm_name}_xsum_with_answers__{llm.model}".replace("/", "_").replace("\\", "_").replace(".", "-") + ".csv"
            }
            # if the csvs already exist, skip the LLM calls and just load the data
            if os.path.exists("results/" + answers_paths["nqopen"]) and not OVERWRITE:
                start_time = time.time()
                print(f"Loading {llm_name} NQ Open answers from csv...")
                nqopen_answers = pd.read_csv("results/" + answers_paths["nqopen"])
                csv_loaded_triggers["nqopen"] = True
                logging.info(f"Time taken to load NQ Open answers: {time.time() - start_time} seconds")
            else:
                start_time = time.time()
                print(f"Generating {llm_name} NQ Open answers...")
                nqopen_llm_answers = evaluation.get_llm_answers(
                    data=nqopen,
                    parallel=True, 
                    temperature=0.0, 
                    logprobs=True,
                    pool=pool,
                    **evaluation.get_NQopen_messages()
                )
                # strip nqopen answers from anything outside the first set of square brackets
                nqopen_llm_answers = [(answer[0], answer[1], answer[2], answer[-1].split('[\'')[1].split('\']')[0].replace('"', '')) for answer in nqopen_llm_answers]
                nqopen_answers = evaluation.create_answers_df(nqopen, nqopen_llm_answers)
                logging.info(f"Time taken to generate NQ Open answers: {time.time() - start_time} seconds")
            if os.path.exists("results/" + answers_paths["xsum"]) and not OVERWRITE:
                start_time = time.time()
                print(f"Loading {llm_name} XSUM answers from csv...")
                xsum_answers = pd.read_csv("results/" + answers_paths["xsum"])
                csv_loaded_triggers["xsum"] = True
                logging.info(f"Time taken to load XSUM answers: {time.time() - start_time} seconds")
            else:
                start_time = time.time()
                print(f"Generating {llm_name} XSUM answers...")
                xsum_llm_answers = evaluation.get_llm_answers(
                    data=xsum,
                    parallel=True, 
                    temperature=0.0, 
                    logprobs=True,
                    pool=pool,
                    **evaluation.get_XSUM_messages()
                )
                xsum_answers = evaluation.create_answers_df(xsum, xsum_llm_answers)
                logging.info(f"Time taken to generate XSUM answers: {time.time() - start_time} seconds")

            # Save answers
            for dataset, path in answers_paths.items():
                if not os.path.exists("results/"):
                    os.makedirs("results/")
                if dataset == "nqopen" and not csv_loaded_triggers["nqopen"]:
                    nqopen_answers.to_csv("results/" + path, index=False)
                elif dataset == "xsum" and not csv_loaded_triggers["xsum"]:
                    xsum_answers.to_csv("results/" + path, index=False)

            # Get hallucination scores
            start_time = time.time()
            nqopen_scores = evaluation.get_hallucination_scores(pool, nqopen_answers, detection_methods, parallel=False)
            xsum_scores = evaluation.get_hallucination_scores(pool, xsum_answers, detection_methods, parallel=False)
            logging.info(f"Time taken to get hallucination scores: {time.time() - start_time} seconds")

            # Save scores
            scores_paths = {
                "nqopen": f"{llm_name}_nqopen_with_scores__{llm.model}".replace("/", "_").replace("\\", "_").replace(".", "-") + ".csv",
                "xsum": f"{llm_name}_xsum_with_scores__{llm.model}".replace("/", "_").replace("\\", "_").replace(".", "-") + ".csv"
            }
            for dataset, path in scores_paths.items():
                if dataset == "nqopen":
                    nqopen_scores.to_csv("results/" + path, index=False)
                elif dataset == "xsum":
                    xsum_scores.to_csv("results/" + path, index=False)
            print(f"Results for {llm_name} saved in results directory.")
