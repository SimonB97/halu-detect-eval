from src.detection.lbhd import LBHD
from src.detection.lm_v_lm import LMvLM
from src.detection.fleek import FLEEK
from src.models.llm import OpenAILlm, TogetherAILlm, BaseLlm
from datasets import load_dataset
import pandas as pd
import os
from multiprocessing import Pool
from ratelimit import RateLimiter
from ratelimit import RateLimitException
import time


class Evaluation:
    def __init__(self, llm: BaseLlm):
        self.llm = llm


    def load_datasets(self):
        nqopen = pd.DataFrame(load_dataset("nq_open", split="validation", cache_dir="datasets/nq_open"))
        xsum = pd.DataFrame(load_dataset("EdinburghNLP/xsum", split="validation", cache_dir="datasets/xsum"))
        return {"nqopen": nqopen, "xsum": xsum}


    def prepare_data(self, datasets: dict[str, pd.DataFrame]):
        nqopen = datasets["nqopen"]
        xsum = datasets["xsum"]

        nqopen["prompt"] = nqopen["question"]
        nqopen = nqopen.drop(["question"], axis=1)
        
        xsum["prompt"] = "Summarize the following article in a single, short sentence:\n\n<article>\n" + xsum["document"] + "\n</article>"
        xsum["answer"] = xsum["summary"]
        xsum = xsum.drop(["document", "summary", "id"], axis=1)

        return {"nqopen": nqopen, "xsum": xsum}
    

    def get_llm_response(self, row, system_message=None, examples=None):
        while True:
            try:
                if examples:
                    messages = examples.copy()
                    messages.append({"role": "user", "content": row["prompt"]})
                    return self.llm.get_response(messages, system_message if system_message else None)[-1]
                else:
                    return self.llm.get_response(row["prompt"], system_message if system_message else None)[-1]
            except RateLimitException:
                time.sleep(1)  # wait for 1 second before retrying


    @RateLimiter(max_calls=50, period=1)  # 50 requests per second
    @RateLimiter(max_calls=5000, period=60)  # 5000 requests per minute
    def get_llm_answers(self, data: pd.DataFrame, system_message: str = None, examples: list[dict] = None, parallel: bool = False):
        if parallel:
            with Pool() as p:
                llm_answers = p.starmap(self.get_llm_response, [(row, system_message, examples) for _, row in data.iterrows()])
        else:
            llm_answers = []
            for index, row in data.iterrows():
                llm_answers.append(self.get_llm_response(row, system_message, examples))
        return llm_answers






if __name__ == "__main__":
    evaluation = Evaluation()
