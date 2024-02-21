from src.detection.lbhd import LBHD
from src.detection.lm_v_lm import LMvLM
from src.detection.fleek import FLEEK
from src.models.llm import OpenAILlm, TogetherAILlm
from datasets import load_dataset
import pandas as pd
import os

# Set Authentication Tokens in .env file!


class Evaluation:
    def __init__(self):
        pass


    def load_datasets(self):
        nqopen = pd.DataFrame(load_dataset("nq_open", split="validation"))
        xsum = pd.DataFrame(load_dataset("EdinburghNLP/xsum", split="validation"))
        return {"nqopen": nqopen, "xsum": xsum}


    def prepare_data(self, datasets: dict[str, pd.DataFrame]):
        nqopen = datasets["nqopen"]
        xsum = datasets["xsum"]

        nqopen["prompt"].rename("question", inplace=True)
        
        xsum["prompt"] = "Summarize the following article in one sentence:\n\n<article>\n" + xsum["document"] + "\n</article>"
        xsum["summary"].rename("answer", inplace=True)

        return {"nqopen": nqopen, "xsum": xsum}

        
    def get_claims(self):
        pass


if __name__ == "__main__":
    evaluation = Evaluation()