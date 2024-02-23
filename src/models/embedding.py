import requests
from dotenv import load_dotenv
from typing import List

load_dotenv()

class TogetherAIEmbedding:
    def __init__(self, bearer_token: str, model: str = "togethercomputer/m2-bert-80M-2k-retrieval", url: str = "https://api.together.xyz/v1/embeddings", debug: bool = False):
        self.bearer_token = bearer_token
        self.model = model
        self.url = url
        self.debug = debug

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "input": text
            }

            try:
                response = requests.post(self.url, json=payload, headers=headers)
                response.raise_for_status()  # This will raise an HTTPError for non-success status codes
                response_data = response.json()
                embedding = [embedding_data["embedding"] for embedding_data in response_data["data"]]
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
                return []  # or handle the error as appropriate for your application
            except Exception as err:
                print(f"Other error occurred: {err}")

            embeddings.append(embedding)

        return embeddings
        
        
