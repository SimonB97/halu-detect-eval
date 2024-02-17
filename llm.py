"""

"""

import requests
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env file

# load bearer token from environment variable
bearer_token = os.getenv("AUTH_BEARER_TOKEN")

url = "https://api.together.xyz/v1/chat/completions"


def get_response(
        message: str,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1,
        top_k: int = 40,
        repetition_penalty: float = 1,
        n: int = 1,
        logprobs: int = 0
) -> requests.Response:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "stop": ["</s>", "[/INST]"],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "n": n,
        "logprobs": logprobs,
        "messages": [
            {
                "role": "user",
                "content": message
            }
    ]
    }

    headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": bearer_token
    }

    response = requests.post(url, json=payload, headers=headers)
    tokens = response.json()["choices"][0]["logprobs"]["tokens"]
    logprobs = response.json()["choices"][0]["logprobs"]["token_logprobs"]

    return tokens, logprobs if logprobs else tokens


if __name__ == "__main__":
    # Example usage
    message = "What is the capital of France? Answer with a single word. The capital of France is "
    response = get_response(message, "mistralai/Mixtral-8x7B-Instruct-v0.1", logprobs=1, max_tokens=1)
    print(response)