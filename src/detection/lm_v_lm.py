"""LM vs LM (LMvLM) hallucination detection method."""

from src.models.llm import BaseLlm


class Examiner:
    def __init__(self, claim, llm: BaseLlm):
        self.message_history = []
        self.claim = claim
        self.llm = llm

    def setup(self):
        Prompts = 'Your goal is to try to verify the correctness of the following claim: <claim>\n{}\n</claim>\n, based on the background information you will gather. \
            To gather this, You will provide short questions whose purpose will be to verify the correctness of the claim, and I will reply to you with the answers to these. \
            Hopefully, with the help of the background questions and their answers, you will be able to reach a conclusion as to whether the claim is correct or possibly incorrect. \
            Please keep asking questions as long as you are yet to be sure regarding the true veracity of the claim. Please start with the first questions.'
        message = {"role": "user", "content": Prompts.format(self.claim)}
        self.message_history.append(message)
        response = self.generate_response()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        
        return response


    def generate_response(self):
        response = self.llm.get_response(self.message_history)[-1]
        return response


class Examinee:
    def __init__(self, entity, claim, llm: BaseLlm):
        self.message_history = []
        self.entity = entity
        self.claim = claim
        self.llm = llm


class LMvLM:
    def __init__(self, examinee: Examinee, examiner: Examiner):
        self.examinee = examinee
        self.examiner = examiner

    def get_hallucination_score(self, response, variants: list[str] = None) -> dict:

        pass

