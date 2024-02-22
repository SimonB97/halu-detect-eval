import pandas as pd
import os
from datasets import load_dataset
from multiprocessing import Pool
from functools import partial


def load_single_dataset(dataset_name, split="validation", cache_dir=None):
    return pd.DataFrame(load_dataset(dataset_name, split=split, cache_dir=cache_dir))

def load_datasets():
    print("Loading datasets...")
    with Pool() as pool:
        datasets = pool.starmap(partial(load_single_dataset), [("nq_open", "validation", "datasets/nq_open"), ("EdinburghNLP/xsum", "validation", "datasets/xsum")])
    return {"nqopen": datasets[0], "xsum": datasets[1]}


def prepare_data(datasets: dict[str, pd.DataFrame]):
        print("Preparing data...")
        nqopen = datasets["nqopen"]
        xsum = datasets["xsum"]

        nqopen["prompt"] = nqopen["question"]
        nqopen = nqopen.drop(["question"], axis=1)
        
        xsum["prompt"] = "Summarize the following article in a single, short sentence:\n\n<article>\n" + xsum["document"] + "\n</article>"
        xsum["answer"] = xsum["summary"]
        xsum = xsum.drop(["document", "summary", "id"], axis=1)

        return {"nqopen": nqopen, "xsum": xsum}