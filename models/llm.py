import os
import requests
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from openai import OpenAI
import numpy as np

load_dotenv()


class BaseLlm(ABC):
    def __init__(self, bearer_token: str, model: str, url: str = None):
        self.bearer_token = bearer_token
        self.model = model
        self.url = url

    @abstractmethod
    def get_response(self, message: str, system_message: str = None, max_tokens: int = 512,
                     temperature: float = 0.0, top_p: float = 1, repetition_penalty: float = 1, 
                     n: int = 1, return_logprobs: bool = False) -> tuple:
        pass


class TogetherAILlm(BaseLlm):
    def get_response(self, message: str, system_message: str = None, max_tokens: int = 512,
                     temperature: float = 0.0, top_p: float = 1, repetition_penalty: float = 1, 
                     n: int = 1, return_logprobs: bool = False) -> tuple:
        url =  self.url if self.url else "https://api.together.xyz/v1/chat/completions"

        messages = [
            {"role": "user", "content": message}
        ]
        if system_message is not None:
            messages.insert(0,
                            {"role": "system", "content": system_message}
            )

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stop": ["</s>", "[/INST]"],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 40,
            "repetition_penalty": repetition_penalty,
            "n": n,
            "logprobs": 0 if not return_logprobs else 1,
            "messages": messages
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": self.bearer_token
        }

        response = requests.post(url, json=payload, headers=headers)
        tokens = response.json()["choices"][0]["logprobs"]["tokens"]
        linear_probabilities = response.json()["choices"][0]["logprobs"]["token_logprobs"] if return_logprobs else None

        if return_logprobs:
            # rescale linear probabilities from -1 to 1 to 0 to 1 (based on assumption for scale of linear probabilities)
            rescaled_linear_probabilities = [(prob + 1) / 2 for prob in linear_probabilities]
            logprobs = [np.log(prob) for prob in rescaled_linear_probabilities]
        else:
            logprobs = None

        # TODO: Not sure on which scale originally returned logprobs are!!! (seems to be 1 (highest) to -1 (lowest) but not sure)

        return tokens, logprobs, linear_probabilities


class OpenAILlm(BaseLlm):
    def get_response(self, message: str, system_message: str = None, max_tokens: int = 512,
                     temperature: float = 0.0, top_p: float = 1, repetition_penalty: float = 1, 
                     n: int = 1, return_logprobs: bool = False) -> tuple:
        url =  "https://api.openai.com/v1" if self.url is None else self.url
        
        messages = [
            {"role": "user", "content": message}
        ]
        if system_message is not None:
            messages.insert(0,
                            {"role": "system", "content": system_message}
            )

        # using requests
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repetition_penalty,
            "n": n,
            "logprobs": return_logprobs,
            "top_logprobs": 1 if return_logprobs else 0,
            "messages": messages
        }
        headers = {
            # "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.bearer_token
        }
        response = requests.post(url + "/chat/completions", json=payload, headers=headers)

        # # using openai python sdk
        # client = OpenAI(api_key=self.bearer_token, base_url=url)
        # response = client.chat.completions.create(
        #     model=self.model,
        #     messages=messages,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     frequency_penalty=repetition_penalty,
        #     n=n,
        #     logprobs=return_logprobs,
        #     top_logprobs=1 if return_logprobs else 0
        # )

        contents = response.json()["choices"][0]["logprobs"]["content"]
        tokens, logprobs = zip(*((content["token"], content["logprob"] if return_logprobs else None) for content in contents))
        tokens = list(tokens)
        logprobs = list(logprobs)

        # Convert logprobs to linear probabilities
        if return_logprobs:
            linear_probabilities = [np.float64(np.round(np.exp(logprob),2)) for logprob in logprobs]
        else:
            linear_probabilities = None

        return tokens, logprobs, linear_probabilities


if __name__ == "__main__":
    # Example usage
    together_bearer_token = os.getenv("TOGETHER_AUTH_BEARER_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    message = "What is the capital of France?"

    llm = OpenAILlm(openai_api_key, "gpt-3.5-turbo")
    oai_response = llm.get_response(message, return_logprobs=True, max_tokens=8)
    print("OpenAI response:\n- Tokens:", oai_response[0], "\n- Logprobs:", oai_response[1], "\n- Linear probabilities:", oai_response[2])
    
    llm = TogetherAILlm(together_bearer_token, "mistralai/Mixtral-8x7B-Instruct-v0.1")
    together_response = llm.get_response(message, return_logprobs=1, max_tokens=8)
    print("TogetherAI response:\n- Tokens:", together_response[0], "\n- Logprobs:", together_response[1], "\n- Linear probabilities:", together_response[2])

    # Plot linear probabilities and logprobs per token curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    tokens_oai = oai_response[0]
    tokens_together = together_response[0]

    ax[0, 0].plot(oai_response[1], label="OpenAI")
    ax[0, 0].set_title("OpenAI Logprobs")
    ax[0, 0].legend()
    ax[0, 0].set_xticks(range(len(tokens_oai)))
    ax[0, 0].set_xticklabels(tokens_oai, fontsize='small')
    ax[0, 0].set_ylabel("Logprob")

    ax[0, 1].plot(together_response[1], label="TogetherAI")
    ax[0, 1].set_title("TogetherAI Logprobs")
    ax[0, 1].legend()
    ax[0, 1].set_xticks(range(len(tokens_together)))
    ax[0, 1].set_xticklabels(tokens_together, fontsize='small')
    ax[0, 1].set_ylabel("Logprob")

    ax[1, 0].plot(oai_response[2], label="OpenAI")
    ax[1, 0].set_title("OpenAI Linear probabilities")
    ax[1, 0].legend()
    ax[1, 0].set_xticks(range(len(tokens_oai)))
    ax[1, 0].set_xticklabels(tokens_oai, fontsize='small')
    ax[1, 0].set_ylabel("Linear probability")

    ax[1, 1].plot(together_response[2], label="TogetherAI")
    ax[1, 1].set_title("TogetherAI Linear probabilities")
    ax[1, 1].legend()
    ax[1, 1].set_xticks(range(len(tokens_together)))
    ax[1, 1].set_xticklabels(tokens_together, fontsize='small')
    ax[1, 1].set_ylabel("Linear probability")

    plt.show()