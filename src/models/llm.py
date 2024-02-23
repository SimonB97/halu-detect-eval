from src.utils.utils import print_json
import os
import requests
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import numpy as np
import json
from ratelimit import limits, sleep_and_retry

load_dotenv()

# 50 calls per second
CALLS = 50
RATE_LIMIT = 1

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    ''' Empty function just to check for calls to API '''


class BaseLlm(ABC):
    def __init__(self, bearer_token: str, model: str, url: str = None, debug: bool = False):
        self.bearer_token = bearer_token
        self.model = model  # model name, e.g. "mistralai/Mistral-7B-Instruct-v0.2" or "gpt-3.5-turbo"
        self.url = url
        self.debug = debug

    @abstractmethod
    def get_response(self, message: str | list[dict], system_message: str = None, max_tokens: int = 512,
                     temperature: float = 0.0, top_p: float = 1, repetition_penalty: float = 1, 
                     n: int = 1, return_logprobs: bool = False, json_mode: bool = False) -> tuple:
        pass


class TogetherAILlm(BaseLlm):
    def __init__(self, bearer_token: str, model: str, url: str = None, debug: bool = False):
        super().__init__(bearer_token, model, url if url else "https://api.together.xyz/v1/chat/completions", debug)

    def get_response(self, message: str | list[dict], system_message: str = None, max_tokens: int = 512,
                     temperature: float = 0.0, top_p: float = 1, repetition_penalty: float = 1, 
                     n: int = 1, return_logprobs: bool = False, json_mode: bool = False) -> tuple:
        check_limit()
        if isinstance(message, list):
            messages = message
            if system_message is not None:
                messages.insert(0,
                                {"role": "system", "content": system_message}
                )
        else:
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
            "messages": messages,
        }
        if json_mode:
            payload["response_format"] = { "type": "json_object" }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": self.bearer_token
        }
        if self.debug: print("DEBUG: TogetherAI request:"), print_json(payload)
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()  # This will raise an HTTPError if the response contains an HTTP error status code
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"Other error occurred: {err}")
        except json.JSONDecodeError:
            print(f"Response could not be parsed as JSON: {response.text}")
        if self.debug: print("DEBUG: TogetherAI response:"), print_json(response.json())

        if return_logprobs:
            tokens = response.json()["choices"][0]["logprobs"]["tokens"]
            linear_probabilities = np.array(response.json()["choices"][0]["logprobs"]["token_logprobs"], dtype=np.float64)

            # rescale linear probabilities from -1 to 1 to 0 to 1 (based on assumption for scale of linear probabilities)
            linear_probabilities = [(prob + 1) / 2 for prob in linear_probabilities]
            logprobs = np.array([np.log(prob) for prob in linear_probabilities], dtype=np.float64)
            full_text = ("".join(tokens))
        else:
            full_text = response.json()["choices"][0]["message"]["content"]
            tokens = None
            logprobs = None
            linear_probabilities = None

        return tokens, logprobs, linear_probabilities, full_text
    
        # TODO: Not sure on which scale originally returned logprobs are!!! (seems to be 1 (highest) to -1 (lowest) but not sure)
            
        
            


class OpenAILlm(BaseLlm):
    def __init__(self, bearer_token: str, model: str, url: str = None, debug: bool = False):
        super().__init__(bearer_token, model, url if url else "https://api.openai.com/v1/chat/completions", debug)

    def get_response(self, message: str | list[dict], system_message: str = None, max_tokens: int = 512,
                     temperature: float = 0.0, top_p: float = 1, repetition_penalty: float = 1, 
                     n: int = 1, return_logprobs: bool = False, json_mode: bool = False) -> tuple:
        check_limit()
        if isinstance(message, list):
            messages = message
            if system_message is not None:
                messages.insert(0,
                                {"role": "system", "content": system_message}
                )
        else:
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
            "stop": "</s>",
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repetition_penalty,
            "n": n,
            "logprobs": return_logprobs,
            "messages": messages
        }
        if return_logprobs:
            payload["top_logprobs"] = 1
        if json_mode:
            payload["response_format"] = { "type": "json_object"}

        headers = {
            # "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.bearer_token
        }
        if self.debug: print("DEBUG: OpenAI request:"), print_json(payload)
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()  # This will raise an HTTPError if the response contains an HTTP error status code
            response_json = response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as err:
            print(f"Other error occurred: {err}")
        except json.JSONDecodeError:
            print(f"Response could not be parsed as JSON: {response.text}")
        if self.debug: print("DEBUG: OpenAI response:"), print_json(response.json())

        try:
            if return_logprobs:
                contents = response.json()["choices"][0]["logprobs"]["content"]
                tokens, logprobs = zip(*((content["token"], np.float64(content["logprob"])) for content in contents))
                tokens = list(tokens)
                logprobs = list(logprobs)
                linear_probabilities = [np.float64(np.exp(logprob)) for logprob in logprobs]
                full_text = ("".join(tokens))
            else:
                full_text = response.json()["choices"][0]["message"]["content"]
                tokens = None
                logprobs = None
                linear_probabilities = None
        except KeyError:
            print(f"Error in response: {response.json()}")
            raise KeyError

        return tokens, logprobs, linear_probabilities, full_text
