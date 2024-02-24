from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckBERTScore
import torch
import numpy as np
import gc

class SelfCheck_NLI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selfcheck_nli = None

    def get_hallucination_score(self, response: str, samples: list[str]) -> float:
        """Calculate hallucination score(s) for the response."""
        scores = self.selfcheck_nli.predict(
            sentences=[response],
            sampled_passages=samples,
        )
        # print(f"DEBUG: SelfCheck NLI scores: {scores} for response: {response} and samples: {samples}")
        gc.collect()
        return {response: {"score": np.mean(scores)}}
    
    def unload_model(self):
        """Unload the model from GPU."""
        self.selfcheck_nli.model.to('cpu')
        self.device = torch.device('cpu')
        torch.cuda.empty_cache()
        gc.collect()

    def load_model(self):
        """Load the model to GPU."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            print("WARNING: No GPU available. Loading model to CPU.")
        print("Loading SelfCheck NLI model...")
        self.selfcheck_nli = SelfCheckNLI(device=self.device)
        gc.collect()
    

class SelfCheck_BERT:
    def __init__(self):
        self.selfcheck_bert = None

    def get_hallucination_score(self, response: str, samples: list[str]) -> float:
        """Calculate hallucination score(s) for the response."""
        scores = self.selfcheck_bert.predict(
            sentences=[response],
            sampled_passages=samples,
        )
        # print(f"DEBUG: SelfCheck BERT scores: {scores} for response: {response} and samples: {samples}")
        gc.collect()
        return {response: {"score": np.mean(scores)}}
    
    def unload_model(self):
        """Unload the model from GPU."""
        self.selfcheck_bert = None
        gc.collect()

    def load_model(self):
        """Load the model to GPU."""
        print("Loading SelfCheck BERT model...")
        self.selfcheck_bert = SelfCheckBERTScore()
        gc.collect()