from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckBERTScore
import torch
import numpy as np

class SelfCheck_NLI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selfcheck_nli = SelfCheckNLI(device=self.device)

    def get_hallucination_score(self, response: str, samples: list[str]) -> float:
        """Calculate hallucination score(s) for the response."""
        scores = self.selfcheck_nli.predict(
            sentences=[response],
            sampled_passages=samples,
        )
        # print(f"DEBUG: SelfCheck NLI scores: {scores} for response: {response} and samples: {samples}")
        return {response: {"score": np.mean(scores)}}
    

class SelfCheck_BERT:
    def __init__(self):
        self.selfcheck_bert = SelfCheckBERTScore()

    def get_hallucination_score(self, response: str, samples: list[str]) -> float:
        """Calculate hallucination score(s) for the response."""
        scores = self.selfcheck_bert.predict(
            sentences=[response],
            sampled_passages=samples,
        )
        # print(f"DEBUG: SelfCheck BERT scores: {scores} for response: {response} and samples: {samples}")
        return {response: {"score": np.mean(scores)}}