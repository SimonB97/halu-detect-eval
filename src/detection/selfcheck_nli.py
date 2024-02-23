from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import torch
import numpy as np

class SelfCheckNLI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selfcheck_nli = SelfCheckNLI(device=self.device)

    def get_hallucination_score(self, response: str, samples: list[str]) -> float:
        """Calculate hallucination score(s) for the response."""
        scores = self.selfcheck_nli.predict(
            sentences=[response],
            sampled_passages=samples,
        )
        return np.mean(scores)  # return the mean of the scores
